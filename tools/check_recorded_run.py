#!/usr/bin/env python3
"""Holds the gate's recorded run to the tree it says it was taken from.

`tests/data/boys_accuracy_gate_run.txt` is a run of the accuracy gate kept as a file: the gate's
own output, which names the version it was taken at and the content of the files it read from,
and prints the delivered figure of every lane and region it swept.
`docs/specification.md`'s printed-run table is a transcription of such a run, and
`tools/check_bound_transcripts.py` holds the table's delivered column to this file. That tie is
between two records - the document and the run - and it cannot see the code, which the file it
reads says plainly: "a run that has gone stale relative to the code satisfies it".

This check is that half, and it reads the two stamp lines through `tools/record_stamp.py`, the
module that states what a digest covers:

  * `accuracy gate, version <v>`, held to the version `include/boys/version.hpp` states - the
    release this tree is, which is what a reader places a run by;
  * `accuracy gate, inputs <group>=<digest> ...`, recomputed from this tree and compared group by
    group. This is the check that holds: the version is bumped by hand, and a source file that
    moved under a stamp nobody bumped is the record this check exists to refuse.

The stamp named a commit until it named content, and the reason for the move is the merge: a pull
request is squash merged, so a commit id written into a tracked artifact is in no clone's history
afterwards, and a checker resolving one refuses the record for a reason about the merge rather
than about the tree. A squash carries the same bytes, so the digest is the identity that survives
it, and this check reads no history at all - it works in a checkout one commit deep.

A group whose digest moved is a finding, because the files the gate reads decide what it prints.
`CMakeLists.txt` is the exception, reported and not gated, for a reason that is a limit of
reading: it decides the flags the gate is compiled under and it decides which tests are in which
target, and a file's digest does not say which of the two a change was. The run's stamp carries
the file's content, so a reader sees it move, and what this check cannot decide is named beside it
for the reader who has the diff in front of them.

The run's own device-lane count is read too: `device lanes : N of M measured on a card`. A run
taken from a build without the device lanes states the shortfall there, and that is a finding
about the run and not about the library - a build with no card answers no question about a device
lane, and what is short is the build the record came from.

What this does not do, and why:
  * it does not run the gate, so it cannot see a difference a build would show that the content
    does not: a compiler that changes its arithmetic between two runs of one tree is a fact about
    the runner, not about this tree, and no recorded run can be held to it. Re-making the run is
    the reading for that, and the reading is a build;
  * it does not read the documents. Whether `docs/specification.md`'s table agrees with the run
    is `tools/check_bound_transcripts.py`'s claim, and the two checks fail apart so that a stale
    record and a mismatched transcription are not reported as one thing.

Usage:
    python tools/check_recorded_run.py
    python tools/check_recorded_run.py --run /tmp/a-run.txt
    python tools/check_recorded_run.py --control

Exit status is 0 when the run names this tree's version and every group of its inputs reads the
same content here, and 1 when one differs, when the stamp names no version or no inputs, when a
cell states `unknown`, when the run states no device-lane count, when the count it states is
short of the lanes its build carries, or when there is no recorded run to read at all. A run
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
RUN = REPO / "tests" / "data" / "boys_accuracy_gate_run.txt"

# The record this check reads, by the name `tools/record_stamp.py` knows its input list by.
RECORD = "gate"

# The two lines the gate stamps itself with, each read by its own spelling rather than by a
# pattern this file invents, so a rewritten stamp is a finding rather than a silently unread run.
VERSION = re.compile(r"^accuracy gate, version[ \t]+(?P<version>\S+)[ \t]*$", re.M)
INPUTS = re.compile(r"^accuracy gate, inputs[ \t]+(?P<inputs>\S+)[ \t]*$", re.M)

# Any line that states the run's own content, whatever cell it names: what a fixture's stamp is
# built over, so a run stamped one way is read whichever way it was stamped.
STAMP_LINE = re.compile(r"^accuracy gate, (version|inputs)\b")

# What the gate prints when it was built where no content digest was passed to it. It names no
# tree, and a run that carries it is one this check cannot hold to anything.
UNKNOWN = "unknown"

# What the run's build reached: `device lanes : 4 of 4 measured on a card`. A run whose first
# number is short of its second is a run of the host lanes alone, and the reason is the build it
# was taken from rather than anything the library does. Read here because a record like that is
# cited for lanes it never touched, and a reader who finds the shortfall in the coverage check
# instead reads it as the library carrying combinations the run does not measure.
DEVICE_LANES = re.compile(r"^device lanes : (?P<measured>\d+) of (?P<carried>\d+) "
                          r"measured on a card[ \t]*$", re.M)

WORK = REPO / "build-recorded-run"


def stamp_module():
    """`tools/record_stamp.py`, loaded from its path rather than copied.

    The digests a run is held to are the ones the tool that wrote its stamp computes, so the two
    readings are one implementation and not two that have to agree.
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
    """The files the gate reads that this working tree holds modified, as git reports them.

    A diagnostic and not a gate, and no history is read for it: what a checkout in CI holds is
    the run's own content, while a developer's uncommitted edits are theirs to judge - and they
    are worth naming, because a digest that differs from the run's most often differs because of
    them.
    """
    paths = tuple(path for group in record_stamp.INPUTS[RECORD].values() for path in group)
    status, dirty, _ = git("status", "--porcelain", "--", *paths)
    if status != 0:
        return []
    return [line[3:].strip() for line in dirty.splitlines() if len(line) > 3]


def read(run_path: pathlib.Path) -> tuple[list[str], list[str]]:
    """The run against the tree: the lines a reader gets, and what is wrong.

    The lines are printed by the caller whatever the verdict, and the findings are what decides
    it. The control below reads its fixtures through this same function, so a control that passes
    is a control over the reading and not over a second copy of it.
    """
    lines = ["the gate's recorded run, held to the version and the inputs it names"]
    findings: list[str] = []

    if not run_path.is_file():
        lines.append(f"  {display(run_path)}: not a file")
        findings.append(
            f"there is no recorded run to read: {display(run_path)} is not a file. A delivered "
            f"figure is what a run of the gate printed, so the document's copy of one is held to "
            f"a run kept as a file, and there is none here to hold it to or to date. Record one: "
            f"build boys-accuracy-gate, run it, and write its output there"
        )
        return lines, findings

    text = run_path.read_text(encoding="utf-8", errors="replace")
    lines.append(f"  {display(run_path)}")

    found = VERSION.search(text)
    named = found.group("version").strip() if found else ""
    lines.append(f"  version: {named or 'none named'}")

    if not named or named == UNKNOWN:
        findings.append(
            f"{display(run_path)} names no version: no line of it reads `accuracy gate, version "
            f"<v>`, which is what places a run in a release. A run that dates itself by nothing "
            f"cannot be told from a stale one, and this check would then be passing over a file "
            f"it cannot place"
        )
    else:
        try:
            tree_version = record_stamp.version(REPO)
        except record_stamp.StampError as error:
            lines.append(f"  this tree: {error}")
            findings.append(
                f"the version this tree states could not be read ({error}), so the run's own "
                f"`version {named}` line is held to nothing. This is the tree and not the record: "
                f"`{record_stamp.VERSION_HEADER}` carries the version as `kVersionString` and a "
                f"configure asserts it against the project's own number"
            )
        else:
            lines.append(f"  this tree: version {tree_version}")
            if named != tree_version:
                findings.append(
                    f"{display(run_path)} names version {named}, and this tree states "
                    f"{tree_version}: the run is an account of another release, and the delivered "
                    f"figures in it are what a release this tree is not printed, so a document "
                    f"held to it is held to figures this tree may not print. Re-make the run - "
                    f"build boys-accuracy-gate in this tree, run it, and write its output to "
                    f"{display(run_path)} - and commit the new record with the change that moved "
                    f"the version"
                )

    found = INPUTS.search(text)
    stated = found.group("inputs").strip() if found else ""

    if not stated or stated == UNKNOWN:
        lines.append(f"  inputs: {stated or 'none named'}")
        findings.append(
            f"{display(run_path)} names no inputs: no line of it reads `accuracy gate, inputs "
            f"<group>=<digest>`, which is what holds the run to the files the gate read. A run "
            f"that names no content cannot be told from one taken before the code moved under it, "
            f"which is the reading this check exists for. Re-make the run from a build of this "
            f"tree"
        )
    else:
        recorded = record_stamp.parse_cell(stated)
        expected = record_stamp.INPUTS[RECORD]
        computed = record_stamp.digests(REPO, RECORD)
        edited = edited_here()

        lines.append(
            "  the gate reads: "
            + ", ".join(path for paths in expected.values() for path in paths)
        )

        for name in record_stamp.gated(RECORD):
            if name not in recorded:
                lines.append(f"    {name}: named by no group of the run's inputs")
                findings.append(
                    f"the run's inputs line holds no {name} group, so the content the gate read "
                    f"there is held to nothing: the line is written by the build and a group it "
                    f"leaves out is one this check cannot compare. Re-make the run from a build "
                    f"of this tree"
                )
            elif recorded[name] != computed[name]:
                lines.append(f"    {name}: recorded {recorded[name]}, this tree {computed[name]}")
                findings.append(
                    f"the content the run was taken from differs in {name}: its stamp states "
                    f"{recorded[name]} and this tree reads {computed[name]}"
                    + (f", over {len(edited)} file(s) the gate reads that are modified here "
                       f"({', '.join(edited[:4])}" + (", ..." if len(edited) > 4 else "") + ")"
                       if edited else "")
                    + f". The run is a record of files this tree does not carry: the delivered "
                    f"figures in it are what the gate printed before the change, so a document "
                    f"held to it is held to figures this tree may not print. Re-make the run - "
                    f"build boys-accuracy-gate in this tree, run it, and write its output to "
                    f"{display(run_path)} - and commit the new record with the change that moved "
                    f"the content"
                )
            else:
                lines.append(f"    {name}: {computed[name]}")

        for name in expected:
            if name in record_stamp.NOT_GATED:
                if name not in recorded:
                    lines.append(
                        "  CMakeLists.txt: named by no group of the run's inputs, so this check "
                        "compared nothing: the line is written by the build and a group it leaves "
                        "out is one this check cannot compare. Re-make the run from a build of "
                        "this tree")
                else:
                    moved = recorded[name] != computed[name]
                    lines.append(
                        "  " + ("CMakeLists.txt differs from the run's stamp: it holds the flags "
                               "the gate compiles under as well as the target lists, and this check "
                               "does not gate it, because a digest does not say which of the two a "
                               "change was - read it, and re-make the run if it moved a flag the "
                               "gate compiles under"
                               if moved else
                               "CMakeLists.txt reads the same content the run's stamp states")
                    )

    # What the run's build reached. The gate states how many of the device lanes its build
    # carries were measured on a card, and a run that measured fewer than it counts is a run of
    # the host lanes alone. The shortfall is the build's and not the library's, so it is refused
    # here, where the sentence can name the artifact and the remedy, rather than surfacing in the
    # coverage check as the library carrying combinations no row measures.
    device = DEVICE_LANES.search(text)

    if device is None:
        findings.append(
            f"{display(run_path)} states no device-lane count: no line of it reads `device lanes "
            f": N of M measured on a card`, which is what the gate prints for the lanes its build "
            f"carried and its card ran. Without it a run taken from a build that carries no "
            f"device lane reads exactly like one that measured them all, and this check cannot "
            f"tell which of the two it is holding. Re-make the run in this tree and commit the "
            f"new record"
        )
        lines.append("  device lanes: none stated")
    else:
        measured = int(device.group("measured"))
        carried = int(device.group("carried"))
        lines.append(f"  device lanes: {measured} of {carried} measured on a card")

        if measured < carried:
            findings.append(
                f"{display(run_path)} records that {measured} of the {carried} device lane(s) "
                f"its build carries were measured on a card. The remaining {carried - measured} "
                f"answer no question here: a combination on one of them is measured by no row of "
                f"this record, and a reader who takes the record's own totals for the library's "
                f"space is reading the host half's run as the whole one. This is the build the "
                f"record was taken from and not the library's space - re-make the run from a "
                f"build with BUILD_CUDA=ON on a host with a usable card, and commit the new record"
            )

    return lines, findings


def stamped(text: str, cells: dict[str, str | None]) -> str:
    """A run with its stamp lines set to these values, and one added where the run carries none.

    A cell whose value is None is taken out. The lines are written where the run carried its own,
    so a fixture differs from a real run in one cell and not in a shape this tool wrote; a run
    carrying no stamp line at all takes them at its end, which is where the gate prints them.
    """
    lines: list[str] = []
    placed = False

    for line in text.splitlines():
        if STAMP_LINE.match(line):
            if not placed:
                lines.extend(f"accuracy gate, {word} {value}"
                             for word, value in cells.items() if value is not None)
                placed = True
            continue
        lines.append(line)

    if not placed:
        lines.extend(f"accuracy gate, {word} {value}"
                     for word, value in cells.items() if value is not None)
    return "\n".join(lines)


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
    """Plant the ways a run stops describing the tree, and require each to be caught.

    The fixtures are built from this tree's own recorded run, so a fixture carries a real run's
    body and differs from it in what the run says about itself. A control that passed would be a
    reading that cannot see the defect it exists for - which is how a record went stale while
    every check over it stayed green.

    The last arm is the other direction: a fixture carrying this tree's own version and this
    tree's own digests, which must read clean. A control whose every arm is caught has not shown
    that the check can pass, and a check that cannot pass is one a reader stops reading.
    """
    source = RUN.read_text(encoding="utf-8", errors="replace") if RUN.is_file() else ""
    text = source if source.strip() else "boys accuracy gate\n"

    WORK.mkdir(parents=True, exist_ok=True)
    clean = record_stamp.digests(REPO, RECORD)
    name, stale = moved_digest()
    differing = dict(clean)
    differing[name] = stale

    # What this tree would stamp its own run with. Every arm below differs from it in one cell,
    # so a finding is the arm's own defect and not a second one read beside it.
    base: dict[str, str | None] = {"version": record_stamp.version(REPO),
                                   "inputs": record_stamp.cell(clean)}

    # Each arm states the reading it is planted for: a control that failed for another reason
    # would be a green control over a check that cannot see this defect.
    arms: list[tuple[str, dict[str, str | None], str | None]] = [
        ("a run with the version line taken out", {"version": None}, "names no version"),
        (f"a build that could not name the version it linked, so its line reads `{UNKNOWN}`",
         {"version": UNKNOWN}, "names no version"),
        ("a run of a version this tree is not", {"version": "0.0.0-planted"}, "another release"),
        ("a run with the inputs line taken out", {"inputs": None}, "names no inputs"),
        (f"a run whose {name} digest is of content this tree does not carry",
         {"inputs": record_stamp.cell(differing)}, "content the run was taken from differs"),
        ("this tree's own version and this tree's own digests", {}, None),
    ]

    failures = 0
    planted = 0

    for index, (what, cells, expected) in enumerate(arms):
        fixture = WORK / f"control-{index}.txt"
        fixture.write_text(stamped(text, {**base, **cells}) + "\n", encoding="utf-8")
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

    # The device-lane floor is a finding of this check and not of the stamp, and nothing else
    # plants it: a run that measured fewer lanes than its build carries reads like a whole one.
    short = "device lanes : 0 of 1 measured on a card"
    floor = WORK / "control-floor.txt"
    floored = stamped(text, base)
    floored = (DEVICE_LANES.sub(short, floored) if DEVICE_LANES.search(floored)
               else floored + "\n" + short)
    floor.write_text(floored + "\n", encoding="utf-8")
    _, floor_findings = read(floor)
    caught = any("measured on a card" in finding and "0 of the 1" in finding
                 for finding in floor_findings)

    print(f"control-floor: a build carrying one device lane that its card did not measure - "
          f"written to {display(floor)}")
    print("  caught" if caught else "  NOT CAUGHT: no finding reads the shortfall")
    failures += 0 if caught else 1
    planted += 1

    if failures:
        print(f"\ncontrol: {failures} of {len(arms) + 1} arm(s) did not read as planted. This "
              f"check cannot see the defect it exists for, so a green run of it says nothing")
        return 1

    print(f"\ncontrol: all {planted} planted defect(s) caught, and the run this tree stamps "
          f"reads clean")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=str(RUN),
                        help="the gate's recorded run: the file that carries the stamp")
    parser.add_argument("--control", action="store_true",
                        help="read a fixture planted in a scratch copy instead of the tree, and "
                             "exit non-zero unless every planted defect is caught")
    args = parser.parse_args()

    if args.control:
        return control()

    lines, findings = read(pathlib.Path(args.run))

    for line in lines:
        print(line)

    if findings:
        print(f"\n{len(findings)} finding(s):\n")
        for finding in findings:
            print(f"STALE: {finding}\n")
        return 1

    print(
        "\nverdict: clean - the recorded run names this tree's version, and every group of the "
        "content it says it read reads the same here: the delivered figures it records are this "
        "tree's gate's"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
