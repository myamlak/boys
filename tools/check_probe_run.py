#!/usr/bin/env python3
"""Holds the option probe's recorded report to the tree it says it was taken from.

`tests/data/boys_option_probe_report.txt` is a run of `boys-option-probe` kept as a file: the
report the tool prints, whose first line stamps the run with the version it was taken at and the
content of the files it read from. `docs/getting-started.md` quotes a measured table out of such
a report, and `tools/check_probe_table.py` holds that quotation to this file. That tie is between
two records - the document and the report - and it cannot see the code: a report taken before a
change to the option surface still satisfies it, while the labels its rows name are options this
tree may not carry. A reader of the document then reads an account of a library that is not this
one.

This check is that half, and it reads the two cells of the stamp through `tools/record_stamp.py`,
the module that states what a digest covers:

  * `version <v>`, held to the version `include/boys/version.hpp` states - the release this tree
    is, which is what a reader places a record by;
  * `inputs <group>=<digest> ...`, recomputed from this tree and compared group by group. This is
    the check that holds: the version is bumped by hand, and a source file that moved under a
    stamp nobody bumped is the record this check exists to refuse.

The stamp named a commit until it named content, and the reason for the move is the merge: a pull
request is squash merged, so a commit id written into a tracked artifact is in no clone's history
afterwards, and a checker resolving one refuses the record for a reason about the merge rather
than about the tree. A squash carries the same bytes, so the digest is the identity that survives
it, and this check reads no history at all - it works in a checkout one commit deep.

A group whose digest moved is a finding, because the files the probe reads decide what it prints.
`CMakeLists.txt` is the exception, reported and not gated, for a reason that is a limit of
reading: it decides the flags the probe is compiled under and it decides which sources are in
which target, and a file's digest does not say which of the two a change was. The report's cell
carries the file's content, so a reader sees it move, and what this check cannot decide is named
beside it for the reader who has the diff in front of them.

What this does not do, and why:
  * it does not run the probe, so it cannot see a difference only a build would show: what an
    option costs is a property of the host and the clock, and no content digest holds a report to
    it. The reading for that is a run, and a run is a measurement;
  * it does not read the report's figures. A cell's fastest entry, its figure and the count of
    options the run could not place behind it are what the probe measured, and they are held to
    the document that quotes them by `tools/check_probe_table.py` and to nothing else.

Usage:
    python3 tools/check_probe_run.py
    python3 tools/check_probe_run.py --report build-probe-record/report.txt
    python3 tools/check_probe_run.py --control

Exit status is 0 when the report names this tree's version and every group of its inputs cell
reads the same content here, and 1 when one differs, when the stamp names no version or no
inputs, when a cell states `unknown`, or when there is no recorded report to read at all. A run
with `--control` exits 0 when every planted defect is caught and 1 when one is not.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import re
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
REPORT = REPO / "tests" / "data" / "boys_option_probe_report.txt"

# The record this check reads, by the name `tools/record_stamp.py` knows its input list by.
RECORD = "probe"

# The words the probe opens its report with. They are what makes a file a probe run rather than
# another closure on disk, and they are read by their own spelling instead of by a pattern this
# file invents, so a rewritten stamp is a finding rather than a silently unread report.
STAMP = "boys option probe | "

# The two cells of that first line, each worded as the emission words it and read wherever in the
# line the emission put it. They are read by their own words and not by their positions, because
# the line already opens with a timestamp other readers key on - a field added before either
# would otherwise read as a missing one. The whitespace inside a cell is not fixed by anything
# this reads, so a report the probe stamped is read whatever its padding.
VERSION = re.compile(r"\|\s*version\s+(?P<version>[^|]*?)\s*(?:\||$)")
INPUTS = re.compile(r"\|\s*inputs\s+(?P<inputs>[^|]*?)\s*(?:\||$)")

# What the probe prints when it was built where no content digest was passed to it. It names no
# tree, and a report that carries it is one this check cannot hold to anything.
UNKNOWN = "unknown"

WORK = REPO / "build-probe-record"


def stamp_module():
    """`tools/record_stamp.py`, loaded from its path rather than copied.

    The digests a report is held to are the ones the tool that wrote its cell computes, so the
    two readings are one implementation and not two that have to agree.
    """
    path = pathlib.Path(__file__).resolve().parent / "record_stamp.py"
    spec = importlib.util.spec_from_file_location("record_stamp", path)
    module = importlib.util.module_from_spec(spec)
    # Registered before it is executed: a class the module decorates looks itself up in
    # `sys.modules` while the module is being built, and a module that is not there fails.
    sys.modules["record_stamp"] = module
    spec.loader.exec_module(module)
    return module


record_stamp = stamp_module()


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


def edited_here() -> list[str]:
    """The files the probe reads that this working tree holds modified, as git reports them.

    A diagnostic and not a gate, and no history is read for it: what a checkout in CI holds is
    the record's own content, while a developer's uncommitted edits are theirs to judge - and
    they are worth naming, because a digest that differs from the record's most often differs
    because of them.
    """
    paths = tuple(path for group in record_stamp.INPUTS[RECORD].values() for path in group)
    status, dirty, _ = git("status", "--porcelain", "--", *paths)
    if status != 0:
        return []
    return [line[3:].strip() for line in dirty.splitlines() if len(line) > 3]


def read(report_path: pathlib.Path) -> tuple[list[str], list[str]]:
    """The report against the tree: the lines a reader gets, and what is wrong.

    The lines are printed by the caller whatever the verdict, and the findings are what decides
    it. The control below reads its fixtures through this same function, so a control that
    passes is a control over the reading and not over a second copy of it.
    """
    lines = ["the option probe's recorded report, held to the version and the inputs it names"]
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

    found = VERSION.search(first)
    named = found.group("version").strip() if found else ""
    lines.append(f"  version: {named or 'none named'}")

    if not named or named == UNKNOWN:
        findings.append(
            f"{display(report_path)} names no version: its first line "
            + (f"carries `version {UNKNOWN}`, which is what a build states when it could not "
               f"name the version it linked"
               if named else
               f"carries no `version` cell, which is what places a run in a release")
            + f". A report that dates itself by nothing cannot be told from a stale one, and "
            f"this check would then be passing over a file it cannot place. The probe states the "
            f"version of the library it linked, so re-make the report from a build of this tree"
        )
    else:
        try:
            tree_version = record_stamp.version(REPO)
        except record_stamp.StampError as error:
            lines.append(f"  this tree: {error}")
            findings.append(
                f"the version this tree states could not be read ({error}), so the report's own "
                f"`version {named}` cell is held to nothing. This is the tree and not the record: "
                f"`{record_stamp.VERSION_HEADER}` carries the version as `kVersionString` and a "
                f"configure asserts it against the project's own number"
            )
        else:
            lines.append(f"  this tree: version {tree_version}")
            if named != tree_version:
                findings.append(
                    f"{display(report_path)} names version {named}, and this tree states "
                    f"{tree_version}: the report is an account of another release, and the labels "
                    f"its rows carry are the options that release served, so a document held to "
                    f"it is held to an option surface this tree may not carry. Re-make the report "
                    f"- build boys-option-probe in this tree, run it, and write its output to "
                    f"{display(report_path)} - and commit the new record with the change that "
                    f"moved the version"
                )

    found = INPUTS.search(first)
    stated = found.group("inputs").strip() if found else ""

    if not stated or stated == UNKNOWN:
        lines.append(f"  inputs: {stated or 'none named'}")
        findings.append(
            f"{display(report_path)} names no inputs: its first line "
            + (f"carries `inputs {UNKNOWN}`, which is what a build states when it was configured "
               f"with no content for the files the probe reads"
               if stated else
               f"carries no `inputs` cell, which is what holds a record to the files it read")
            + f". A report that names no content cannot be told from one taken before the code "
            f"moved under it, which is the reading this check exists for. Re-make the report from "
            f"a build of this tree"
        )
        return lines, findings

    recorded = record_stamp.parse_cell(stated)
    expected = record_stamp.INPUTS[RECORD]
    computed = record_stamp.digests(REPO, RECORD)
    edited = edited_here()

    lines.append(
        "  the probe reads: "
        + ", ".join(path for paths in expected.values() for path in paths)
    )

    for name in record_stamp.gated(RECORD):
        if name not in recorded:
            lines.append(f"    {name}: named by no group of the report's cell")
            findings.append(
                f"the report's inputs cell holds no {name} group, so the content the probe read "
                f"there is held to nothing: the cell is written by the build and a group it "
                f"leaves out is one this check cannot compare. Re-make the report from a build "
                f"of this tree"
            )
        elif recorded[name] != computed[name]:
            lines.append(f"    {name}: recorded {recorded[name]}, this tree {computed[name]}")
            findings.append(
                f"the content the report was taken from differs in {name}: its cell states "
                f"{recorded[name]} and this tree reads {computed[name]}"
                + (f", over {len(edited)} file(s) the probe reads that are modified here "
                   f"({', '.join(edited[:4])}" + (", ..." if len(edited) > 4 else "") + ")"
                   if edited else "")
                + f". The report is an account of files this tree does not carry, and the labels "
                f"its rows name are the options that build served, so a document held to it is "
                f"held to an option surface this tree may not have. Re-make the report - build "
                f"boys-option-probe in this tree, run it, and write its output to "
                f"{display(report_path)} - and commit the new record with the change that moved "
                f"the content"
            )
        else:
            lines.append(f"    {name}: {computed[name]}")

    for name in expected:
        if name in record_stamp.NOT_GATED:
            moved = name in recorded and recorded[name] != computed[name]
            lines.append(
                "  " + ("CMakeLists.txt differs from the report's cell: it holds the flags the "
                       "probe compiles under as well as the target lists, and this check does "
                       "not gate it, because a digest does not say which of the two a change "
                       "was - read it, and re-make the report if it moved a flag the probe "
                       "measures under"
                       if moved else
                       "CMakeLists.txt reads the same content the report's cell states")
            )

    return lines, findings


def stamped(header: str, cells: dict[str, str | None]) -> str:
    """A report's first line with the named cells set to these values, and one added where the
    line carries none. A cell whose value is None is taken out.

    The fixture's line is built out of the line this tree's report opens with, cell by cell, so
    that a fixture differs from a real report in one cell and not in a shape this tool wrote.
    """
    parts = [part.strip() for part in header.split(" | ") if part.strip()]
    kept = [part for index, part in enumerate(parts)
            if index == 0 or part.split(" ", 1)[0] not in cells]

    for word, value in cells.items():
        if value is None:
            continue
        at = next((index for index, part in enumerate(kept)
                   if part.split(" ", 1)[0] == "verdict"), len(kept))
        kept.insert(at, f"{word} {value}")
    return " | ".join(kept)


def moved_digest() -> tuple[str, str]:
    """One gated group's digest as a tree whose content moved reads it.

    The group's own files are copied under the scratch directory and one byte of the first of
    them is changed, so the value is a digest of content that differs from this tree's by that
    edit - the defect this check exists to see, and not a typo of the right shape.
    """
    name = record_stamp.gated(RECORD)[0]
    paths = record_stamp.INPUTS[RECORD][name]
    relative = record_stamp.files(REPO, paths)
    scratch = WORK / "moved"

    if scratch.exists():
        shutil.rmtree(scratch)
    for entry in relative:
        target = scratch / entry
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / entry, target)

    edited = scratch / relative[0]
    edited.write_bytes(edited.read_bytes() + b"\n")
    return name, record_stamp.digest(scratch, paths)


def control() -> int:
    """Plant the ways a report stops describing the tree, and require each to be caught.

    The fixtures are built from this tree's own recorded report, so a fixture carries a real
    report's body and differs from it in what the run says about itself. A control that passed
    would be a reading that cannot see the defect it exists for - which is how a report went
    stale while every check over it stayed green.

    The last arm is the other direction: a fixture carrying this tree's own version and this
    tree's own digests, which must read clean. A control whose every arm is caught has not shown
    that the check can pass, and a check that cannot pass is one a reader stops reading.
    """
    source = REPORT.read_text(encoding="utf-8", errors="replace") if REPORT.is_file() else ""
    header = (source.splitlines()[0] if source.splitlines()
              else f"{STAMP}started 1970-01-01T00:00:00Z | verdict CANNOT DETERMINE")
    body = "\n".join(source.splitlines()[1:])

    WORK.mkdir(parents=True, exist_ok=True)
    clean = record_stamp.digests(REPO, RECORD)
    name, stale = moved_digest()
    differing = dict(clean)
    differing[name] = stale

    # What this tree would stamp its own report with. Every arm below differs from it in one
    # cell, so a finding is the arm's own defect and not a second one read beside it.
    base: dict[str, str | None] = {"version": record_stamp.version(REPO),
                                   "inputs": record_stamp.cell(clean)}

    # Each arm states the reading it is planted for: a control that failed for another reason
    # would be a green control over a check that cannot see this defect.
    arms: list[tuple[str, dict[str, str | None], str | None]] = [
        ("a stamp with the version cell taken out", {"version": None}, "names no version"),
        (f"a build that could not name the version it linked, so its cell reads `{UNKNOWN}`",
         {"version": UNKNOWN}, "names no version"),
        ("a report of a version this tree is not", {"version": "0.0.0-planted"},
         "another release"),
        ("a stamp with the inputs cell taken out", {"inputs": None}, "names no inputs"),
        (f"a report whose {name} digest is of content this tree does not carry",
         {"inputs": record_stamp.cell(differing)}, "content the report was taken from differs"),
        ("this tree's own version and this tree's own digests", {}, None),
    ]

    failures = 0
    planted = 0

    for index, (what, cells, expected) in enumerate(arms):
        fixture = WORK / f"control-{index}.txt"
        fixture.write_text(stamped(header, {**base, **cells}) + "\n" + body, encoding="utf-8")
        _, findings = read(fixture)

        print(f"control {index}: {what} - written to {display(fixture)}")

        if expected is None:
            caught = not findings
            print("  clean" if caught else f"  NOT CLEAN: {findings[0][:96]}")
        else:
            planted += 1
            caught = any(expected in finding for finding in findings)
            print("  caught" if caught else f"  NOT CAUGHT: no finding reads `{expected}`")

        failures += 0 if caught else 1

    if failures:
        print(f"\ncontrol: {failures} of {len(arms)} arm(s) did not read as planted. This check "
              f"cannot see the defect it exists for, so a green run of it says nothing")
        return 1

    print(f"\ncontrol: all {planted} planted defect(s) caught, and the report this tree stamps "
          f"reads clean")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=str(REPORT),
                        help="the probe's recorded report: the file that carries the stamp")
    parser.add_argument("--control", action="store_true",
                        help="read a fixture planted in a scratch copy instead of the tree, and "
                             "exit non-zero unless every planted defect is caught")
    args = parser.parse_args()

    if args.control:
        return control()

    lines, findings = read(pathlib.Path(args.report))

    for line in lines:
        print(line)

    if findings:
        print(f"\n{len(findings)} finding(s):\n")
        for finding in findings:
            print(f"STALE: {finding}\n")
        return 1

    print(
        "\nverdict: clean - the recorded report names this tree's version, and every group of "
        "the content it says it read reads the same here: the labels its rows carry are this "
        "tree's probe's"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
