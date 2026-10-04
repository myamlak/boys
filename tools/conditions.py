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
    args = parser.parse_args()

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
