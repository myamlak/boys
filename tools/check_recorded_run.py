#!/usr/bin/env python3
"""Holds the gate's recorded run to the tree it says it was taken from.

`tests/data/boys_accuracy_gate_run.txt` is a run of the accuracy gate kept as a
file: the gate's own output, which names the revision it was taken at and prints
the delivered figure of every lane and region it swept.
`docs/specification.md`'s printed-run table is a transcription of such a run,
and `tools/check_bound_transcripts.py` holds the table's delivered column to
this file. That tie is between two records - the document and the run - and it
cannot see the code, which the file it reads says plainly: "a run that has gone
stale relative to the code satisfies it".

This check is that half. It reads the revision out of the recorded run and asks
the repository what has changed since, over the files that decide what the gate
prints: the library's sources and headers, the gate and its reference header,
and the committed reference grid it measures against. A file that moved is a
finding, and the fix is a new run rather than an edit: the recorded file is one
build's output on one host, and no other tree's run can be made to say what this
tree's gate prints.

A commit that touches none of those files moves nothing the gate reads - the
same build would print the same figures - so the record is still this tree's and
this check passes, saying how many commits it looked past and over what.

The run's own device-lane count is read too: `device lanes : N of M measured on a
card`. A run taken from a build without the device lanes states the shortfall
there, and that is a finding about the run and not about the library - a build
with no card answers no question about a device lane, and what is short is the
build the record came from.

THE BUILD DESCRIPTION IS REPORTED AND NOT GATED, and the reason is a limit of
reading rather than a judgement: `CMakeLists.txt` decides the flags the gate is
compiled under, and it also decides which tests are in which target, and a file
diff does not say which of the two a change was. Failing on every edit to it
would fail on the addition of a test to a target this gate is not in, and
passing on every edit would pass on a change to the arithmetic options. So this
check says what it saw - the file moved, or did not - and names what it cannot
decide, on every run, for the reader who has the diff in front of them.

What this does not do, and why:
  * it does not run the gate, so it cannot see a difference a build would show
    that the file list does not: a compiler that changes its arithmetic between
    two runs of one revision is a fact about the runner, not about this tree,
    and no recorded run can be held to it. Re-making the run is the reading for
    that, and the reading is a build;
  * it does not read the documents. Whether `docs/specification.md`'s table
    agrees with the run is `tools/check_bound_transcripts.py`'s claim, and the
    two checks fail apart so that a stale record and a mismatched transcription
    are not reported as one thing.

Usage:
    python tools/check_recorded_run.py
    python tools/check_recorded_run.py --run /tmp/a-run.txt --revision main

Exit status is 0 when every file the gate reads is unchanged since the revision
the run names, and 1 when one is, when the run names no revision, when the
revision is not in this repository's history, or when there is no recorded run
to read at all.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
RUN = REPO / "tests" / "data" / "boys_accuracy_gate_run.txt"

# What the gate prints from: the library it links (sources and headers), the
# gate itself and the reference header it includes, and the committed reference
# grid it measures against. The recorded run is not among them - the record is
# not an input to what it records - and neither is anything under tools/ or
# docs/, which the gate never reads.
INPUTS = (
    "src",
    "include",
    "tests/boys_accuracy_gate.cpp",
    "tests/boys_gate_reference.hpp",
    "tests/data/boys_accuracy_gate_reference.csv",
)

# The build description, which is an input of a different kind: it decides the
# flags the gate is compiled under, and this check cannot read a flag out of it.
# A CMakeLists that adds a test to another target and a CMakeLists that moves
# the arithmetic options this gate compiles under differ in the diff and not in
# the file, so a rule that failed on both would fail on the moving of a test
# list, and one that passed on both would pass on a flag change. It is reported
# on every run instead, with what this check cannot decide stated where the
# reader of the log can act on it.
BUILD_DESCRIPTION = "CMakeLists.txt"

# The revision line the gate prints, the same one `check_bound_transcripts.py`
# reads: `accuracy gate, revision <rev>`. A value of `unknown` is what the gate
# prints when it was configured where git was not found, and it names no tree.
REVISION = re.compile(r"^accuracy gate, revision[ \t]+(?P<rev>\S+)[ \t]*$", re.M)

# What the run's build reached: `device lanes : 4 of 4 measured on a card`. A run
# whose first number is short of its second is a run of the host lanes alone, and
# the reason is the build it was taken from rather than anything the library does.
# Read here because a record like that is cited for lanes it never touched, and a
# reader who finds the shortfall in the coverage check instead reads it as the
# library carrying combinations the run does not measure.
DEVICE_LANES = re.compile(r"^device lanes : (?P<measured>\d+) of (?P<carried>\d+) "
                          r"measured on a card[ \t]*$", re.M)


def display(path: pathlib.Path) -> str:
    """A path as this tool names it: relative to the repository when it is inside it."""
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def git(*args: str) -> tuple[int, str, str]:
    """git in this repository, as (exit status, stdout, stderr).

    stdout is not trimmed: `git status --porcelain` puts a space before the
    status letter, and trimming the first line of it would eat a character of the
    first path. The callers that want one value trim it themselves.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(REPO), *args],
            capture_output=True,
            text=True,
        )
    except OSError as error:
        return 127, "", f"git could not be run: {error}"
    return proc.returncode, proc.stdout, proc.stderr.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=str(RUN),
                        help="the gate's recorded run: the file that names the revision")
    parser.add_argument("--revision", default="HEAD",
                        help="the revision to hold the run to (default: the working tree's HEAD)")
    args = parser.parse_args()

    run_path = pathlib.Path(args.run)
    findings: list[str] = []
    edited: list[str] = []

    lines = [f"the gate's recorded run, held to {args.revision}"]
    if not run_path.is_file():
        findings.append(
            f"there is no recorded run to read: {display(run_path)} is not a file. A delivered "
            f"figure is what a run of the gate printed, so the document's copy of one is held to a "
            f"run kept as a file, and there is none here to hold it to or to date. Record one: "
            f"build boys-accuracy-gate, run it, and write its output there"
        )
        lines.append(f"  {display(run_path)}: not a file")
    else:
        text = run_path.read_text(encoding="utf-8", errors="replace")
        lines.append(f"  {display(run_path)}")
        match = REVISION.search(text)
        if match is None:
            findings.append(
                f"{display(run_path)} names no revision: no line of it reads `accuracy gate, "
                f"revision <rev>`, which is what the gate prints and what dates the run. A run that "
                f"dates itself by nothing cannot be told from a stale one, and this check would "
                f"then be passing over a file it cannot place in time"
            )
            lines.append("  revision: none named")
        else:
            revision = match.group("rev")
            lines.append(f"  revision: {revision}")
            status, resolved, error = git("rev-parse", "--verify", "--quiet", f"{revision}^{{commit}}")
            resolved = resolved.strip()
            if status != 0 or not resolved:
                findings.append(
                    f"the run names revision {revision}, and this repository cannot resolve it"
                    + (f" ({error})" if error else "")
                    + ". A record whose revision is not in the history being checked cannot be told "
                    "from a stale one: fetch the revision the run names (a shallow checkout has "
                    "only the tip commit), or record a run of a revision this tree carries"
                )
                lines.append("  resolved: no - the revision is not in this repository's history")
            else:
                status, changed, error = git("diff", "--name-only", resolved, args.revision,
                                             "--", *INPUTS)
                if status != 0:
                    findings.append(
                        f"git could not compare {revision} with {args.revision}: "
                        f"{error or 'no reason given'}"
                    )
                    lines.append("  compared: no")
                else:
                    moved = [line for line in changed.splitlines() if line.strip()]
                    status, count, _ = git("rev-list", "--count", f"{revision}..{args.revision}")
                    commits = count.strip() if status == 0 else "an unknown number of"
                    status, build_moved, _ = git("diff", "--name-only", resolved, args.revision,
                                                 "--", BUILD_DESCRIPTION)
                    lines.append(
                        f"  the gate reads: {', '.join(INPUTS)}"
                    )
                    lines.append(
                        "  " + (f"{BUILD_DESCRIPTION} moved since the run: it is the flags as well "
                                "as the target lists, and this check cannot tell the two apart - "
                                "read it, and re-make the run if it moved a flag the gate compiles "
                                "under"
                                if build_moved.strip() else
                                f"{BUILD_DESCRIPTION} is unchanged since the run")
                    )
                    # A record names a revision, and what a working tree holds is
                    # not always that revision. This is not a finding: a checkout
                    # in CI is the revision named, and a developer's uncommitted
                    # edits are theirs to judge. It is said out loud because the
                    # figures the run records are the committed tree's, and a
                    # reader comparing them against a local build would otherwise
                    # be comparing two different trees without being told.
                    status, dirty, _ = git("status", "--porcelain", "--", *INPUTS)
                    edited = [line[3:].strip() for line in dirty.splitlines() if len(line) > 3]
                    if edited:
                        lines.append(
                            f"  the working tree: {len(edited)} file(s) the gate reads are "
                            f"modified here ({', '.join(edited[:4])}"
                            + (", ..." if len(edited) > 4 else "")
                            + f"); the record describes {revision} and not these edits"
                        )
                    if not moved:
                        lines.append(
                            f"  changed since: none, over {commits} commit(s) between {revision} "
                            f"and {args.revision}"
                        )
                    else:
                        lines.append(
                            f"  changed since, over {commits} commit(s) between {revision} and "
                            f"{args.revision}: {len(moved)} file(s)"
                        )
                        for name in moved[:12]:
                            lines.append(f"    {name}")
                        if len(moved) > 12:
                            lines.append(f"    ... and {len(moved) - 12} more")
                        findings.append(
                            f"the recorded run was taken at {revision}, and {len(moved)} file(s) the "
                            f"gate reads from have changed since ({moved[0]}"
                            + (f" and {len(moved) - 1} more" if len(moved) > 1 else "")
                            + f"). The run is a record of a different build: the delivered figures "
                            f"in it are what the gate printed before the change, so a document held "
                            f"to it is held to figures this tree may no longer print. Re-make the "
                            f"run - build boys-accuracy-gate at {args.revision}, run it, and write "
                            f"its output to {display(run_path)} - and commit the new record with the "
                            f"change that moved the figures"
                        )

        # What the run's build reached. The gate states how many of the device
        # lanes its build carries were measured on a card, and a run that measured
        # fewer than it counts is a run of the host lanes alone. The shortfall is
        # the build's and not the library's, so it is refused here, where the
        # sentence can name the artifact and the remedy, rather than surfacing in
        # the coverage check as the library carrying combinations no row measures.
        device = DEVICE_LANES.search(text)
        if device is None:
            findings.append(
                f"{display(run_path)} states no device-lane count: no line of it reads `device "
                f"lanes : N of M measured on a card`, which is what the gate prints for the lanes "
                f"its build carried and its card ran. Without it a run taken from a build that "
                f"carries no device lane reads exactly like one that measured them all, and this "
                f"check cannot tell which of the two it is holding. Re-make the run at "
                f"{args.revision} and commit the new record"
            )
            lines.append("  device lanes: none stated")
        else:
            measured = int(device.group("measured"))
            carried = int(device.group("carried"))
            lines.append(f"  device lanes: {measured} of {carried} measured on a card")
            if measured < carried:
                findings.append(
                    f"{display(run_path)} records that {measured} of the {carried} device lane(s) "
                    f"its build carries were measured on a card. The remaining "
                    f"{carried - measured} answer no question here: a combination on one of them is "
                    f"measured by no row of this record, and a reader who takes the record's own "
                    f"totals for the library's space is reading the host half's run as the whole "
                    f"one. This is the build the record was taken from and not the library's space - "
                    f"re-make the run from a build with BUILD_CUDA=ON on a host with a usable card, "
                    f"and commit the new record"
                )

    for line in lines:
        print(line)
    if findings:
        print(f"\n{len(findings)} finding(s):\n")
        for finding in findings:
            print(f"STALE: {finding}\n")
        return 1

    print(
        "\nverdict: clean - the recorded run names a revision this tree carries, and no file the "
        "gate prints from has changed since: the delivered figures it records are this tree's "
        "gate's"
        + (f", over the {len(edited)} uncommitted edit(s) named above, which the record does not "
           f"describe" if edited else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
