#!/usr/bin/env python3
r"""Evaluate the eleven merge conditions and report the state of the instrument.

The merge conditions are absolute statements about the final artefact. Their history is that each one
was, at some point, called met on the strength of a local check promoted to a global claim, and the
promotion was invisible because the claim read the same however shallow the check behind it was.

This tool exists to make that promotion impossible to perform silently. It does not estimate. It runs
the command that evaluates each condition and prints that command, this revision, and its exit code:

    python3 tools/conditions.py --report

A condition with no command is printed as NO EVALUATOR, and the count of those is the honest answer to
"how far away is the merge": the distance cannot be measured for a condition that nothing evaluates,
and a sentence claiming otherwise would be a forecast wearing the clothes of a measurement.

**What this tool cannot do, stated here because the limit is the point.** It evaluates conditions that
reduce to a query over what the tree declares. A defect in the DECLARATION - a class that should exist
and is not named anywhere, an axis the row format cannot express, a checker that exits zero because it
asks the wrong question - is invisible to every command here. That class is not bounded and no tool
closes it. The purpose of these commands is to make the knowable checks cheap so that attention is
left for the ones that are not.

Usage:

    python3 tools/conditions.py --report          # every condition, with its command and result
    python3 tools/conditions.py --only 1,5        # a subset, for a quick read
    python3 tools/conditions.py --list            # the conditions and their commands, running nothing
"""

import argparse
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
PY = sys.executable

# One record per condition. `cmd` is the command that evaluates it, run from the repository root; a
# condition with `cmd is None` has no evaluator and says why. `timeout` is a real bound, and a
# timeout is reported as a timeout rather than as a pass.
CONDITIONS = [
    {
        "n": 1,
        "title": "Complete option space (every class complete on all three legs)",
        "cmd": [PY, "tools/combination_matrix.py", "--report",
                ".claude/lane-status/probes/host-report.txt"],
        "covers": "per class (device, precision, shape): the combinations that can be instantiated, "
                  "how many the run measured, and which are missing by name — the owner's condition "
                  "read as one enumeration instead of three separate claims",
        "does_not_cover": "the device half and the default leg, which the tool does not read yet; a "
                          "run that checks no class exits 1 rather than passing quietly",
        "timeout": 300,
    },
    {
        "n": 1,
        "title": "Complete option space",
        "cmd": [PY, "tools/check_class_surface.py", "--check"],
        "covers": "every (precision, shape) class the header's own enumerations imply is served by an "
                  "entry or states a reason",
        "does_not_cover": "the DEVICE class surface, which this command does not walk; and a class "
                          "that ought to be declared and is not declared anywhere",
        "timeout": 60,
    },
    {
        "n": 1,
        "title": "Complete option space (the served/refused enumeration)",
        "cmd": [PY, "tools/status.py"],
        "covers": "one closed arithmetic per option space, with the served, refused and unaccounted "
                  "counts summing to the space's size",
        "does_not_cover": "whether the space being closed is the space that ought to exist",
        "timeout": 900,
    },
    {
        "n": 2,
        "title": "No silent substitution",
        "cmd": [PY, "tools/check_seam_macros.py", "--check"],
        "covers": "the seam macros a build reads, against the entries that read them",
        "does_not_cover": "a substitution performed by a body rather than by a macro",
        "timeout": 120,
    },
    {
        "n": 2,
        "title": "No silent substitution (the default rows)",
        "cmd": [PY, "tools/check_default_rows_have_entries.py", "--check"],
        "covers": "every row of the default table against the entries that resolve to it",
        "does_not_cover": "a row that names an axis value the guarantee table does not carry",
        "timeout": 120,
    },
    {
        "n": 3,
        "title": "Everything enabled measured",
        "cmd": [PY, "tools/check_recorded_run.py"],
        "covers": "the recorded gate run against the files the gate prints from",
        "does_not_cover": "a member enabled since the run was recorded",
        "timeout": 300,
    },
    {
        "n": 4,
        "title": "Verified regeneration",
        "cmd": None,
        "why": "the check re-derives every coefficient and takes about four hours; it is run "
               "deliberately before the merge and its transcript is committed, not run here",
        "how_to_run": "python3 tools/gen_boys_coefficients.py --check",
    },
    {
        "n": 5,
        "title": "CI green on the final revision",
        "cmd": ["gh", "pr", "checks", "8", "--repo", "myamlak/boys"],
        "covers": "the checks GitHub ran on the pull request's current head",
        "does_not_cover": "the device path: every leg sets BUILD_CUDA=OFF and the device bounds are a "
                          "documented local gate",
        "timeout": 120,
        "exit_means": "zero when every check has passed; a pending check is not a pass",
    },
    {
        "n": 6,
        "title": "No documented claim the code contradicts",
        "cmd": [PY, "tools/check_doc_arithmetic.py", "--strict"],
        "covers": "every arithmetic claim in the documentation that this tool can derive",
        "does_not_cover": "a claim of sameness, identity or mechanism - which is the class that has "
                          "produced the two confirmed false claims found this week",
        "timeout": 300,
    },
    {
        "n": 7,
        "title": "Usable consumer documentation",
        "cmd": [PY, "tools/check_doc_transcripts.py", "--check"],
        "covers": "every quoted run in the documentation against what the program printed",
        "does_not_cover": "usability, which is not a property a command can read",
        "timeout": 300,
    },
    {
        "n": 8,
        "title": "House rules",
        "cmd": [PY, "tools/check_conditional_spans.py", "--check"],
        "covers": "a construct a conditional opens being closed inside the same branch",
        "does_not_cover": "the format, Doxygen and vocabulary gates, which run in CI and locally",
        "timeout": 300,
    },
    {
        "n": 8,
        "title": "House rules (the entry surface)",
        "cmd": [PY, "tools/check_host_entry_bijection.py", "--check"],
        "covers": "every declared host name against the shapes the table places it in, both directions",
        "does_not_cover": "the same question on the device surface, which its twin check covers",
        "timeout": 120,
    },
    {
        "n": 8,
        "title": "House rules (the probe measures every declared entry)",
        "cmd": [PY, "tools/check_probe_measures_every_entry.py"],
        "covers": "every entry the host headers declare against the entries the option probe calls, "
                  "so an entry nothing times cannot read as a class the library does not serve",
        "does_not_cover": "whether a called entry is called at every combination of the axes, which "
                          "the probe's own closure accounts for in its report",
        "timeout": 120,
    },
    {
        "n": 10,
        "title": "Defaults according to design",
        "cmd": None,
        "why": "the evaluator is the round trip: emit a seam from a probe run, configure a build "
               "against it, and require the suite green. It needs a quiet machine and a probe run "
               "for keeps, and no command here can stand in for those",
        "how_to_run": "the probe emits a seam; configure -DBOYS_BUILD_DEFAULTS=<seam>; run ctest",
    },
    {
        "n": 11,
        "title": "Documentation whose examples are compiled and checked",
        "cmd": [PY, "tools/check_example_transcripts.py", "--check", "--examples-dir",
                "build-win/Release"],
        "covers": "every number the documentation quotes against what the built example printed",
        "does_not_cover": "an example the documentation does not quote",
        "timeout": 300,
    },
]


# The freeze: what must hold before an option probe may run for keeps. A probe run over a space
# that is still moving produces defaults the next commit invalidates, so the run is held until
# every one of these is true. Five are checkable here; the sixth is not, and says so rather than
# being quietly counted as met.
FREEZE = [
    {
        "name": "no uncommitted work in the tree a probe would be run from",
        "cmd": ["git", "status", "--porcelain"],
        "empty": True,
        "timeout": 60,
    },
    {
        # The HOST space closes from the accuracy gate's recorded run, which exists before a probe
        # does. The DEVICE and PROBE spaces close from the probes' own reports, which are what a
        # probe run PRODUCES - requiring them here would be a gate that can never open, because the
        # thing it waits for is the thing it guards. This ran the whole of status.py once and was
        # blocked by exactly that circularity.
        "name": "the host option space closes (from the gate's recorded run)",
        "cmd": [PY, "tools/check_host_space_closes.py"],
        "empty": False,
        "timeout": 900,
    },
    {
        "name": "the host class surface is served",
        "cmd": [PY, "tools/check_class_surface.py", "--check"],
        "empty": False,
        "timeout": 60,
    },
    {
        "name": "the device class surface is carried",
        "cmd": [PY, "tools/check_device_class_surface.py", "--check"],
        "empty": False,
        "timeout": 60,
    },
    {
        # Five entries were declared, served and never timed until this was asked: the probe's row
        # list was short of the surface, so five classes reported "no option of this precision and
        # shape produced a figure on this run" - a sentence about the probe that reads as one about
        # the library - and their seam rows were written from the file's own names. A run over that
        # space produces defaults for classes nothing measured, which is why the question is a
        # freeze condition and not only a house rule.
        "name": "the option probe calls every entry the host surface declares",
        "cmd": [PY, "tools/check_probe_measures_every_entry.py"],
        "empty": False,
        "timeout": 120,
    },
    {
        # The owner's condition, as one gate: every class (device, precision, shape) complete on
        # all three legs - its possible combinations implemented, probed, and a default of its own.
        # The tool does not read the device half or the default leg yet, and it exits 1 when it
        # checked no class rather than passing quietly, so this reads short until the work is done
        # rather than reading clean on the strength of what it does cover.
        "name": "every class is complete: its possible combinations implemented, probed, defaulted",
        "cmd": [PY, "tools/combination_matrix.py", "--report",
                ".claude/lane-status/probes/host-report.txt"],
        "empty": False,
        "timeout": 300,
    },
    {
        # The entry half of the same question: an entry the surface declares and the probe never
        # calls is a class whose row reads "no option of this precision and shape produced a
        # figure" - a sentence about the probe taken for one about the library.
        "name": "the option probe measures every combination a class can be instantiated at",
        "cmd": [PY, "tools/check_class_combinations.py", "--report",
                ".claude/lane-status/probes/host-report.txt", "--no-compile"],
        "empty": False,
        "timeout": 120,
    },
    {
        "name": "the recorded run is current",
        "cmd": [PY, "tools/check_recorded_run.py"],
        "empty": False,
        "timeout": 300,
    },
    {
        "name": "no lane is editing the tree",
        "cmd": None,
        "why": "a lane mid-edit is a space that is still moving, and no command can see a worktree "
               "another agent holds. Count the lanes and say the number: the freeze is not met "
               "while any of them is running.",
    },
    # The owner's ruling is that a combination may be unsupported only for a very serious
    # fundamental reason, and that finding whether one exists is the work - so the condition is not
    # "every refusal states a reason" but "there is no refusal that should not be". That is already
    # read above: the host entry requires the certified count to equal the space's own size, and the
    # arithmetic's second term is the refused-and-owed count, so a single refusal that is not
    # fundamental fails that entry by making certified short of total. The device and probe spaces
    # close from the probes' own reports, which do not exist yet, so their refusal counts are read
    # after their runs rather than here.
]


def freeze() -> int:
    """Report the freeze conditions, each with the command that decided it."""
    print("the freeze: what must hold before a probe runs for keeps\n")

    unmet = 0
    for condition in FREEZE:
        if condition["cmd"] is None:
            print(f"[NOT CHECKABLE] {condition['name']}")
            print(f"    {condition['why']}\n")
            unmet += 1
            continue

        try:
            completed = subprocess.run(condition["cmd"], cwd=REPO, capture_output=True, text=True,
                                       timeout=condition["timeout"], errors="replace")
        except (subprocess.TimeoutExpired, FileNotFoundError) as failure:
            print(f"[FAIL        ] {condition['name']}")
            print(f"    command : {' '.join(condition['cmd'])}")
            print(f"    result  : {failure}\n")
            unmet += 1
            continue

        output = completed.stdout.strip()
        # An "empty" condition is met by producing no output at all; every other is met by exiting
        # zero. Both are read, because a command that exits zero while printing a complaint is the
        # shape of failure this project has been bitten by.
        if condition["empty"]:
            met = completed.returncode == 0 and output == ""
            detail = "no output" if output == "" else output.splitlines()[0]
        else:
            met = completed.returncode == 0
            detail = (output.splitlines()[-1] if output else "(no output)")

        print(f"[{'MET' if met else 'UNMET':<12}] {condition['name']}")
        print(f"    command : {' '.join(condition['cmd'])}")
        print(f"    result  : exit {completed.returncode}, {detail[:110]}\n")
        unmet += 0 if met else 1

    print(f"{len(FREEZE) - unmet} of {len(FREEZE)} freeze condition(s) met")
    if unmet:
        print("THE SPACE IS STILL MOVING. A probe run now produces defaults the next commit "
              "invalidates, and the run is spent.")
    return 1 if unmet else 0


def mark(state: str) -> str:
    return {"pass": "PASS", "fail": "FAIL", "no-evaluator": "NO EVALUATOR",
            "timeout": "TIMEOUT"}[state]


def run(condition: dict) -> tuple[str, str, float]:
    """Run one condition's command. Returns (state, detail, seconds)."""
    if condition["cmd"] is None:
        return "no-evaluator", condition["why"], 0.0

    started = time.monotonic()

    try:
        completed = subprocess.run(condition["cmd"], cwd=REPO, capture_output=True, text=True,
                                   timeout=condition["timeout"], errors="replace")
    except subprocess.TimeoutExpired:
        return "timeout", f"exceeded {condition['timeout']}s", time.monotonic() - started
    except FileNotFoundError as missing:
        return "no-evaluator", f"the command is not installed here: {missing}", 0.0

    seconds = time.monotonic() - started
    tail = [line for line in (completed.stdout + completed.stderr).splitlines() if line.strip()]

    return ("pass" if completed.returncode == 0 else "fail"), (tail[-1] if tail else "(no output)"), seconds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="run every evaluator and report")
    parser.add_argument("--list", action="store_true", help="print the conditions and their commands")
    parser.add_argument("--only", default="", help="a comma-separated list of condition numbers")
    parser.add_argument("--freeze", action="store_true",
                        help="report whether the option space has stopped moving, which is the "
                             "gate a probe run for keeps waits on")
    args = parser.parse_args()

    if args.freeze:
        return freeze()

    wanted = {int(part) for part in args.only.split(",") if part.strip()} if args.only else None
    selected = [c for c in CONDITIONS if wanted is None or c["n"] in wanted]

    if args.list:
        for condition in selected:
            command = " ".join(condition["cmd"]) if condition["cmd"] else "NO EVALUATOR"
            print(f"{condition['n']:>2}  {condition['title']}\n    {command}")
        return 0

    revision = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip()
    print(f"the eleven merge conditions, evaluated at {revision}")
    print("each row names the command that produced it; an uncommanded claim is not a row\n")

    states: dict[str, int] = {}
    for condition in selected:
        state, detail, seconds = run(condition)
        states[state] = states.get(state, 0) + 1
        print(f"[{mark(state):<12}] {condition['n']:>2}  {condition['title']}")
        print(f"    command : {' '.join(condition['cmd']) if condition['cmd'] else '(none)'}")
        print(f"    covers  : {condition['covers'] if condition['cmd'] else condition['why']}")
        if condition["cmd"]:
            print(f"    does not: {condition['does_not_cover']}")
            print(f"    result  : {detail}   ({seconds:.1f}s)")
        else:
            print(f"    how     : {condition['how_to_run']}")
        print()

    evaluators = states.get("pass", 0) + states.get("fail", 0) + states.get("timeout", 0)
    print(f"{evaluators} of {len(selected)} row(s) have an evaluator; "
          f"{states.get('no-evaluator', 0)} do not")
    print(f"  PASS {states.get('pass', 0)}   FAIL {states.get('fail', 0)}   "
          f"TIMEOUT {states.get('timeout', 0)}   NO EVALUATOR {states.get('no-evaluator', 0)}")
    print("\nThe distance to a merge is not this table. This table is what can be measured; a "
          "sentence about the distance is a forecast, and every forecast given before this table "
          "existed was wrong in the same direction.")

    return 1 if (states.get("fail", 0) or states.get("timeout", 0)) else 0


if __name__ == "__main__":
    sys.exit(main())
