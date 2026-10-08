#!/usr/bin/env python
"""The completion status: one closed arithmetic per option space, or the reason it does not close.

WHY THIS EXISTS. A status composed from memory is a summary of the spaces that came to mind, and a
space nobody thought about is absent from that summary rather than visible in it as a hole. The
host space can close exactly - 504 + 0 + 168 + 0 + 0 = 672 - with the device's classes in no
arithmetic at all, and a question about them then finds the gap in the enumeration, not in the prose.

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

# Suffixes that are not a report whatever they contain. A closure is printed by a program to its
# stdout and filed as text; anything packed, archived or linked is an artefact about a tree and not
# a measurement of one.
BINARY_SUFFIXES = {".tar", ".gz", ".tgz", ".zip", ".7z", ".exe", ".dll", ".obj", ".lib", ".pdb",
                   ".png", ".jpg", ".pdf"}

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

# The library's own record of a device run it withdrew. The phrase is the withdrawal; the run it
# names is the card and the date the sentence carries (`a Quadro T1000, on 2026-10-04`). Both are
# read from the committed text rather than written here, so the run that retires this line retires
# it by itself and a second withdrawn run needs no edit to this file.
DEFAULTS_SEAM = os.path.join(REPO, "include", "boys", "boys_build_defaults.hpp")
WITHDRAWN_PROSE = re.compile(r"withdrawn rather than published")
WITHDRAWN_RUN = re.compile(
    r"-\s+(?:a|an|the)?\s*([A-Za-z][A-Za-z0-9\-]*(?:\s+[A-Za-z0-9\-]+)*),\s*on\s+(\d{4}-\d{2}-\d{2})")

# The run's own statement that its instrument was not steady: a fixed work read by the same clock
# the entries were, so a pass flagged here is a pass whose clock moved by more than the alarm.
CANARY_WIDE = re.compile(
    r"(\d+)\s+of\s+(\d+)\s+pass\(es\)\s+ran\s+with\s+the\s+canary's\s+own\s+runs\s+wider\s+than\s+"
    r"the\s+([\d.]+)%\s+alarm")


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
    # IS IT CURRENT: asked of the checker that owns the question, not decided here. This used to
    # compare the run's revision stamp with HEAD and call any mismatch STALE, which is a cruder
    # question than the one that matters - whether a file the gate PRINTS FROM has changed - and it
    # reported a false NOT CLOSED on a record that tools/check_recorded_run.py calls clean, because
    # two ordinary commits had landed since. A wrong NOT CLOSED is as bad as a wrong CLOSED.
    stale = ""
    try:
        checked = subprocess.run([sys.executable,
                                  os.path.join(REPO, "tools", "check_recorded_run.py")],
                                 capture_output=True, text=True, timeout=180, cwd=REPO)
        if checked.returncode != 0:
            stale = (f"   NOT CURRENT: tools/check_recorded_run.py exits {checked.returncode} - "
                     f"the run is {revision[:9]}")
    except Exception:
        stale = f"   UNCHECKED: the recorded run is {revision[:9]} and the checker could not be run"

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


def seam_prose(path: str) -> str:
    """A seam header's prose, with each line's comment prefix folded to one space.

    The sentence that names a withdrawn run wraps, and every wrapped line opens with the comment
    marker, so the card and the date sit either side of `///` in the file's own text. A pattern
    written against the sentence as it reads finds neither. This reads the sentence as it reads,
    not as the file wraps it.
    """
    return re.sub(r"^\s*//[/!<]?\s?", " ", read(path), flags=re.M)


def withdrawn_run(path: str, text: str) -> list[str]:
    """The lines to print when `path` is a device run the library's own text records as withdrawn.

    A closure that sums is an arithmetic about a run. Whether the RUN stands is a different
    question, and the closure does not answer it: its verdict is computed over member states and
    never looks at the instrument. The tree does answer it - the device defaults state that a run
    on a named card and date "found every pass wider than its own canary's alarm and was withdrawn
    rather than published" - and a space closed on that run is closed on evidence the project has
    withdrawn, which is the host space's stale-record defect one level down.

    Identity is the CARD the ruling names, and deliberately not the date. Every device closure on
    disk is a run of that card from those two days, every one of them is canary-wide, and every one
    of them would print CLOSED from the moment the newest file was removed - the fallback is the
    same defect, not a sound run. The date the ruling carries is printed with it, to say which run
    is meant; it is not made a test, because a file's mtime describes the file and not the run.
    """
    ruling = seam_prose(DEFAULTS_SEAM)
    found = WITHDRAWN_PROSE.search(ruling)
    if not found:
        return []
    named = WITHDRAWN_RUN.findall(ruling[:found.start()])
    if not named:
        # The ruling is there and the run it names is not readable. Refusing loudly is the whole
        # point: a parse that quietly stops matching would put this space back to closing on a
        # withdrawn run the first time the sentence was reworded.
        return [f"   UNREAD: {os.path.basename(DEFAULTS_SEAM)} records a device run withdrawn and "
                f"this reads no card and date from it, so whether {os.path.basename(path)} is that "
                f"run is unknown and the space is not closed on an unread ruling"]
    on = [f"{card}, on {date}" for card, date in named if card in text]
    if not on:
        return []

    lines = [f"   WITHDRAWN: {os.path.basename(path)} is a run on {on[0]}, which the device "
             f"defaults ({os.path.basename(DEFAULTS_SEAM)}) record as withdrawn rather than "
             f"published"]
    canary = CANARY_WIDE.search(text)
    if canary:
        lines.append(f"   its own canary block reads {canary.group(1)} of {canary.group(2)} "
                     f"pass(es) ran wider than the {canary.group(3)}% alarm")
    return lines


def device_space() -> tuple[str, list[str], bool]:
    """The device option space, from the probe report that carries its closure arithmetic.

    The probe prints the closure as its last block: the space's total read off the library's own
    tables (`BoysDeviceOptions().size()` x `kDeviceRungs.size()`), the states each member can be in,
    the arithmetic summing them, and a verdict. A log written before that block existed carries no
    closure at all - which is a fact about the log, not about the library - so the search is for the
    block, and the absence of it is reported as the absence of a RUN rather than of a space.

    The closure is one question - do the member states sum to the space - and it is not the whole
    of the one that matters. A closure can sum correctly over a run the project has withdrawn, and
    this tool printed exactly that as CLOSED: the run behind the 324 was a Quadro T1000 whose own
    canary block flagged every pass, which the device defaults record as withdrawn rather than
    published. So the run's standing is asked too, of the library's own text, and a withdrawn run
    refuses the closure the same way a stale host run does.
    """
    carrying = []
    # The tree's own reports, and not a scratch directory. A closure read out of a scratch path
    # is a closure of whatever run last wrote there: this tool read a superseded device run from
    # a scratch path while the committed report said something else, and printed the scratch
    # one. The device probe's report is committed beside the host probe's, and the newest file on
    # disk is not a fact about this tree.
    for directory in (os.path.join(REPO, "tests", "data"),):
        for root, _dirs, names in os.walk(directory):
            for name in names:
                path = os.path.join(root, name)
                # A REPORT IS TEXT. A packed tree, an archive, a binary or an object file can carry
                # the same words a report prints - a `base-*.tar` of a revision contains the source
                # that names the option table - and reading one as a closure reports a space from an
                # artifact that measured nothing. This tool did exactly that: it read a control tar
                # as the device closure and reported the block "present but not in the shape this
                # reads", which is a true statement about a file that is not a run.
                if os.path.splitext(path)[1].lower() in BINARY_SUFFIXES:
                    continue
                text = read(path)
                # The WHOLE closure block, not the phrase. A lane's write-up quotes "the arithmetic:"
                # while discussing a run, and reading that as a closure is exactly the wrong read this
                # tool exists to make impossible: it printed a bookkeeping file as a measurement and
                # reported the block "present but not in the shape this reads".
                if has_closure(text) and "BoysDeviceOptions" in text:
                    carrying.append(path)

    if not carrying:
        return ("ABSENT: no committed report carries a whole device closure",
                ["  the probe prints a closure (MEMBERS / the arithmetic / the space's own total /",
                 "  the verdict) and no report under tests/data carries all four, so the device",
                 "  space cannot be closed from what this tree carries. Owed: the probe's report,",
                 "  committed - tests/data/boys_device_probe_report.txt."], False)

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
    # IS IT A RUN THAT STANDS, which the closure does not say. The probe's verdict is computed over
    # member states and never over the instrument, so the arithmetic can agree on a run whose own
    # canary block says the clock moved under every pass. The project's ruling on such a run is in
    # the library's text; reading it here is what keeps this line from being a success not read.
    withdrawn = withdrawn_run(newest, text)
    if withdrawn:
        lines += withdrawn
        closed = False
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
    walks - two arithmetics over the same library, neither derivable from the other.

    The space is read from the probe's own committed report, whose first line is the probe's own
    stamp (`boys option probe | started ... | verdict ...`). That stamp is what identifies it: a
    closure on disk that carries no stamp does not say which space its arithmetic is over. Where no
    report carries the stamp, the space is printed ABSENT and the report it waits for is named
    there, rather than left out of the status entirely.

    Nothing is read off another closure in its place, because **a closure read the wrong way is
    worse than one not read**: the point of this tool is that a wrong arithmetic is never printed
    as a right one, and a space missing from a status is indistinguishable from a space that is
    complete.
    """
    # IDENTIFIED, and only because the probe stamps its own output. The first line of a run is
    # `boys option probe | started <ts> | finished <ts> | seed <n> | verdict <v>`, which is the stamp
    # this space lacked: three other closures sit on disk - the host probe's cell cross, the option
    # book's axes cross and the region-B book, two of them also totalling 720 - and none of them says
    # which space its arithmetic is over. Reading "the newest whole closure" printed the option
    # book's 720 as this space's answer once; this reads the probe's own header instead.
    runs = []
    # The tree's own reports. The probe's run is committed at
    # tests/data/boys_option_probe_report.txt, and a run read out of a scratch directory is a run
    # of whatever last wrote there - this tool printed a lane's superseded host run once, and
    # "newest file on disk" is not a fact about this tree.
    for root, _dirs, names in os.walk(os.path.join(REPO, "tests", "data")):
        for name in names:
            path = os.path.join(root, name)
            try:
                if os.path.getsize(path) > 8 * 1024 * 1024:
                    continue
                with open(path, encoding="utf-8", errors="replace") as handle:
                    first = handle.readline()
            except OSError:
                continue
            if first.startswith("boys option probe | started"):
                runs.append(path)

    if not runs:
        return ("ABSENT: no committed report is a probe run (none starts with the probe's own header)",
                ["  the probe stamps its first line `boys option probe | started ... | verdict ...`;",
                 "  no report under tests/data carries it, so this space is printed as a hole rather",
                 "  than left out of the status entirely. Owed: the probe's report, committed."],
                False)

    newest = max(runs, key=os.path.getmtime)
    with open(newest, encoding="utf-8", errors="replace") as handle:
        first = handle.readline().rstrip()
    text = read(newest)
    found = ARITHMETIC_WITH_NUMBERS.search(text)
    total = found.group(0).rsplit("=", 1)[1].strip() if found else "?"

    lines = [f"  source                 {os.path.basename(newest)}",
             f"  the run's own header   {first[:120]}"]
    closed = False
    if found:
        body = found.group(0).split("the arithmetic:")[1].strip()
        lhs, rhs = body.split("=")
        parts = [int(n) for n in re.findall(r"\d+", lhs)]
        lines.append(f"  the run's own arithmetic {body.strip()}")
        # The probe's own states, exactly as the gate's: a part that does not sum is open, and the
        # 144 device cells are a state of their own - an entry of that lane is a call on a CUDA
        # device, which no cell of this host can run whatever the library documents for it. That is
        # the same shape as the host space's "not runnable here", and it does not open the space.
        closed = sum(parts) == int(rhs.strip())
        lines.append(f"  unchanged                 the arithmetic sums" if closed else
                     f"  DISAGREES with the total: {' + '.join(map(str, parts))} = {sum(parts)}")
    else:
        lines.append("  the file is a probe run but carries no arithmetic this reads")
    # A probe run is a measurement of one machine at one moment, and the header names when.
    return (f"{total} member(s) measured", lines, closed)


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
