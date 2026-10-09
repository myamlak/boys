#!/usr/bin/env python3
"""Holds the option probe's recorded report to the tree it says it was taken from.

`tests/data/boys_option_probe_report.txt` is a run of `boys-option-probe` kept as a file: the
report the tool prints, whose first line stamps the run with the revision it was taken at and the
verdict it reached. `docs/getting-started.md` quotes a measured table out of such a report, and
`tools/check_probe_table.py` holds that quotation to this file. That tie is between two records -
the document and the report - and it cannot see the code: a report taken before a change to the
option surface still satisfies it, while the labels its rows name are options this tree may not
carry. A reader of the document then reads an account of a library that is not this one.

This check is that half. It reads the revision out of the report's first line and asks the
repository what has changed since, over the files that decide what the probe prints: the
library's sources and headers, and the probe's own driver. A file that moved is a finding, and
the reading for it is a new run rather than an edit: the report is one build's measurement on one
host, and no other tree's run can be made to say what this tree's probe prints.

A commit that touches none of those files moves nothing the probe reads - the same build would
print the same report - so the record is still this tree's and this check passes, saying how many
commits it looked past and over what.

THE BUILD DESCRIPTION IS REPORTED AND NOT GATED, and the reason is a limit of reading rather than
a judgement: `CMakeLists.txt` decides the flags a target is compiled under, and it also decides
which sources are in which target, and a file diff does not say which of the two a change was.
The probe measures the options a build carries, so the flags are an input of substance to its
answer; the file is printed as moved or unmoved on every run, with what this check cannot decide
named beside it, for the reader who has the diff in front of them.

What this does not do, and why:
  * it does not run the probe, so it cannot see a difference only a build would show: what an
    option costs is a property of the host and the clock, and no file listing holds a report to
    it. The reading for that is a run, and a run is a measurement;
  * it does not read the report's figures. A cell's fastest entry, its figure and the count of
    options the run could not place behind it are what the probe measured, and they are held to
    the document that quotes them by `tools/check_probe_table.py` and to nothing else.

Usage:
    python3 tools/check_probe_run.py
    python3 tools/check_probe_run.py --report build-probe-record/report.txt --revision main
    python3 tools/check_probe_run.py --control

Exit status is 0 when every file the probe reads from is unchanged since the revision the report
names, and 1 when one is, when the report names no revision, when the revision is not in this
repository's history, or when there is no recorded report to read at all. A run with `--control`
exits 0 when every planted defect is caught and 1 when one is not.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
REPORT = REPO / "tests" / "data" / "boys_option_probe_report.txt"

# What the probe prints from: the library it links (sources and headers) and the driver that calls
# it. The report is not among them - the record is not an input to what it records - and neither
# is anything under tools/, docs/ or tests/, which the probe never reads.
INPUTS = (
    "src",
    "include",
    "benchmarks/boys_option_probe.cpp",
)

# The build description, an input of a different kind: it holds the flags the probe is compiled
# under, and this check cannot read a flag out of it. Reported, never gated.
BUILD_DESCRIPTION = "CMakeLists.txt"

# The words the probe opens its report with. They are what makes a file a probe run rather than
# another closure on disk, and they are read by their own spelling instead of by a pattern this
# file invents, so a rewritten stamp is a finding rather than a silently unread report.
STAMP = "boys option probe | "

# The revision cell of that first line: the cell worded `revision`, wherever in the line the
# emission put it. It is read by its own word and not by its position, because the line already
# opens with a timestamp other readers key on - a field added before this one would otherwise
# read as a missing revision. The whitespace inside a cell is not fixed by anything this reads,
# so a report the probe stamped is read whatever its padding.
REVISION = re.compile(r"\|\s*revision\s+(?P<rev>[^|]*?)\s*(?:\||$)")

# What the probe prints when it was built where no revision was passed to it. It names no tree,
# and a report that carries it is one this check cannot place in time.
UNKNOWN = "unknown"

WORK = REPO / "build-probe-record"


def display(path: pathlib.Path) -> str:
    """A path as this tool names it: relative to the repository when it is inside it."""
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def git(*args: str) -> tuple[int, str, str]:
    """git in this repository, as (exit status, stdout, stderr).

    stdout is not trimmed: `git status --porcelain` puts a space before the status letter, and
    trimming the first line of it would eat a character of the first path.
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


def read(report_path: pathlib.Path, revision: str) -> tuple[list[str], list[str]]:
    """The report against the tree: the lines a reader gets, and what is wrong.

    The lines are printed by the caller whatever the verdict, and the findings are what decides
    it. The control below reads its fixtures through this same function, so a control that
    passes is a control over the reading and not over a second copy of it.
    """
    lines = [f"the option probe's recorded report, held to {revision}"]
    findings: list[str] = []

    if not report_path.is_file():
        lines.append(f"  {display(report_path)}: not a file")
        findings.append(
            f"there is no recorded report to read: {display(report_path)} is not a file. The "
            f"figures a document quotes from the probe are what a run printed, so the document's "
            f"copy of them is held to a run kept as a file, and there is none here to hold or to "
            f"date. Record one: build boys-option-probe, run it, and write its output there"
        )
        return lines, findings

    text = report_path.read_text(encoding="utf-8", errors="replace")
    first = text.splitlines()[0] if text.splitlines() else ""
    lines.append(f"  {display(report_path)}")

    if not first.startswith(STAMP):
        lines.append("  stamp: none")
        findings.append(
            f"{display(report_path)} opens with no probe stamp: its first line does not read "
            f"`{STAMP}`, which is what the probe prints and what makes a file a run of it rather "
            f"than another report on disk. A file this check cannot recognize is one it cannot "
            f"date, and a report it cannot date is held to nothing"
        )
        return lines, findings

    found = REVISION.search(first)
    named = found.group("rev").strip() if found else ""

    if not named or named == UNKNOWN:
        lines.append(f"  revision: {named or 'none named'}")
        findings.append(
            f"{display(report_path)} names no revision: its first line "
            + (f"carries `revision {UNKNOWN}`, which is what the probe prints when it was built "
               f"where no revision was passed to it"
               if named else
               f"carries no `revision` cell, which is what dates a run")
            + f". A report that dates itself by nothing cannot be told from a stale one, and this "
            f"check would then be passing over a file it cannot place in time. The probe writes "
            f"the revision its build was configured with, so re-make the report from a build of "
            f"this tree"
        )
        return lines, findings

    lines.append(f"  revision: {named}")

    status, resolved, error = git("rev-parse", "--verify", "--quiet", f"{named}^{{commit}}")
    resolved = resolved.strip()

    if status != 0 or not resolved:
        lines.append("  resolved: no - the revision is not in this repository's history")
        findings.append(
            f"the report names revision {named}, and this repository cannot resolve it"
            + (f" ({error})" if error else "")
            + ". A record whose revision is not in the history being checked cannot be told from "
            "a stale one: fetch the revision the report names (a shallow checkout has only the "
            "tip commit), or record a run of a revision this tree carries"
        )
        return lines, findings

    status, changed, error = git("diff", "--name-only", resolved, revision, "--", *INPUTS)

    if status != 0:
        lines.append("  compared: no")
        findings.append(
            f"git could not compare {named} with {revision}: {error or 'no reason given'}"
        )
        return lines, findings

    moved = [line for line in changed.splitlines() if line.strip()]
    status, count, _ = git("rev-list", "--count", f"{named}..{revision}")
    commits = count.strip() if status == 0 else "an unknown number of"
    status, build_moved, _ = git("diff", "--name-only", resolved, revision, "--",
                                 BUILD_DESCRIPTION)

    lines.append(f"  the probe reads: {', '.join(INPUTS)}")
    lines.append(
        "  " + (f"{BUILD_DESCRIPTION} moved since the run: it holds the flags the probe compiles "
                "under as well as the target lists, and this check cannot tell the two apart - "
                "read it, and re-make the report if it moved a flag the probe measures under"
                if build_moved.strip() else
                f"{BUILD_DESCRIPTION} is unchanged since the run")
    )

    # What a working tree holds is not always the revision a report names. This is not a finding:
    # a checkout in CI is the revision named, and a developer's uncommitted edits are theirs to
    # judge. It is said out loud because the figures the report records are the committed tree's,
    # and a reader comparing them against a local build would otherwise be comparing two trees
    # without being told.
    status, dirty, _ = git("status", "--porcelain", "--", *INPUTS)
    edited = [line[3:].strip() for line in dirty.splitlines() if len(line) > 3]

    if edited:
        lines.append(
            f"  the working tree: {len(edited)} file(s) the probe reads are modified here "
            f"({', '.join(edited[:4])}" + (", ..." if len(edited) > 4 else "")
            + f"); the report describes {named} and not these edits"
        )

    if not moved:
        lines.append(
            f"  changed since: none, over {commits} commit(s) between {named} and {revision}"
        )
        return lines, findings

    lines.append(
        f"  changed since, over {commits} commit(s) between {named} and {revision}: "
        f"{len(moved)} file(s)"
    )

    for name in moved[:12]:
        lines.append(f"    {name}")

    if len(moved) > 12:
        lines.append(f"    ... and {len(moved) - 12} more")

    findings.append(
        f"the recorded report was taken at {named}, and {len(moved)} file(s) the probe reads "
        f"from have changed since ({moved[0]}"
        + (f" and {len(moved) - 1} more" if len(moved) > 1 else "")
        + f"). The report is an account of a different build: the labels its rows carry are the "
        f"options that build served, so a document held to it is held to an option surface this "
        f"tree may not have. Re-make the report - build boys-option-probe at {revision}, run it, "
        f"and write its output to {display(report_path)} - and commit the new record with the "
        f"change that moved the figures"
    )
    return lines, findings


def stamped(header: str, revision: str | None) -> str:
    """A report's first line with its revision cell set to `revision`, and one added where it
    carries none.

    The fixture's line is built out of the line this tree's report opens with, cell by cell, so
    that a fixture differs from a real report in one cell and not in a shape this tool wrote.
    """
    parts = [part for part in header.split(" | ") if not part.strip().startswith("revision ")]

    if revision is None:
        return " | ".join(parts)

    at = next((index for index, part in enumerate(parts) if part.strip().startswith("verdict")),
              len(parts))
    parts.insert(at, f"revision {revision}")
    return " | ".join(parts)


def control() -> int:
    """Plant the ways a report stops describing the tree, and require each to be caught.

    The fixtures are built from this tree's own recorded report, so a fixture carries a real
    report's body and differs from it in what the run says about itself. A control that passed
    would be a reading that cannot see the defect it exists for - which is how a report went
    stale while every check over it stayed green.
    """
    source = REPORT.read_text(encoding="utf-8", errors="replace") if REPORT.is_file() else ""
    header = (source.splitlines()[0] if source.splitlines()
              else f"{STAMP}started 1970-01-01T00:00:00Z | verdict CANNOT DETERMINE")
    body = "\n".join(source.splitlines()[1:])

    status, oldest, _ = git("rev-list", "--max-parents=0", "HEAD")
    root = oldest.strip().splitlines()[0] if status == 0 and oldest.strip() else ""

    # Each arm states the reading it is planted for: a control that failed for another reason
    # would be a green control over a check that cannot see this defect.
    arms: list[tuple[str, str | None, str]] = [
        ("a stamp with the revision cell taken out", None, "names no revision"),
        (f"a build that was passed no revision, so its cell reads `{UNKNOWN}`", UNKNOWN,
         "names no revision"),
        ("a revision this repository's history does not carry", "0f0f0f0f0f0f",
         "cannot resolve it"),
    ]

    # The arm a shallow checkout cannot plant: it needs a revision old enough for an input to
    # have moved past it, and a checkout whose history is one commit deep has none. The arm is
    # reported as skipped with that reason rather than counted as one that passed.
    skipped = ""

    if root:
        status, ahead, _ = git("diff", "--name-only", root, "HEAD", "--", *INPUTS)
        if status == 0 and ahead.strip():
            arms.append((f"a report of revision {root[:12]}, which the probe's inputs have moved "
                         f"past", root, "have changed since"))
        else:
            skipped = (f"the arm planted from revision {root[:12]} is skipped: nothing under "
                       f"{', '.join(INPUTS)} differs between it and HEAD, so a fixture naming it "
                       f"is not a report the inputs have moved past")
    else:
        skipped = ("the arm planted from the oldest revision is skipped: this checkout's history "
                   "does not name one")

    WORK.mkdir(parents=True, exist_ok=True)
    failures = 0

    for index, (what, revision, expected) in enumerate(arms):
        fixture = WORK / f"control-{index}.txt"
        fixture.write_text(stamped(header, revision) + "\n" + body, encoding="utf-8")
        _, findings = read(fixture, "HEAD")
        caught = any(expected in finding for finding in findings)

        print(f"control {index}: {what} - written to {display(fixture)}")
        print(f"  {'caught' if caught else 'NOT CAUGHT'}"
              + ("" if caught else f": no finding reads `{expected}`"))
        failures += 0 if caught else 1

    if skipped:
        print(f"  skipped: {skipped}")

    if failures:
        print(f"\ncontrol: {failures} of {len(arms)} planted defect(s) were not caught. This "
              f"check cannot see the defect it exists for, so a green run of it says nothing")
        return 1

    print(f"\ncontrol: all {len(arms)} planted defect(s) caught")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=str(REPORT),
                        help="the probe's recorded report: the file that names the revision")
    parser.add_argument("--revision", default="HEAD",
                        help="the revision to hold the report to (default: the working tree's HEAD)")
    parser.add_argument("--control", action="store_true",
                        help="read a fixture planted in a scratch copy instead of the tree, and "
                             "exit non-zero unless every planted defect is caught")
    args = parser.parse_args()

    if args.control:
        return control()

    lines, findings = read(pathlib.Path(args.report), args.revision)

    for line in lines:
        print(line)

    if findings:
        print(f"\n{len(findings)} finding(s):\n")
        for finding in findings:
            print(f"STALE: {finding}\n")
        return 1

    print(
        "\nverdict: clean - the recorded report names a revision this tree carries, and no file "
        "the probe reads from has changed since: the labels its rows carry are this tree's probe's"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
