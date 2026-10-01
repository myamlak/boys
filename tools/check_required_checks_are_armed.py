#!/usr/bin/env python3
r"""Check that every check this repository relies on is armed by a required CI leg.

`.github/required-checks.txt` is the list of check names branch protection requires on main, and
`.github/workflows/ci.yml` is what reports those names. Both are written by hand and have different
consumers, and the ways they can disagree - with each other, and with the checks the tree actually
carries - are all silent ones:

  * a name in `required-checks.txt` that no leg renders. Branch protection then holds a merge open
    waiting for a status check no job ever posts, and the file reads as though that leg were gated.
    The file's own header states the trap: "a name that matches nothing is silently unenforced".
  * a checker under `tools/` that no required leg runs, because its step was deleted, or edited so
    that it no longer invokes the tool, or scoped by an `if:` to a leg outside the list. The leg
    still runs, still prints and still passes, and a red one stops no merge.
  * a `--leg` argument that names a rendering of no leg of its own job. The comparison it feeds
    does not fail on an unknown leg, so a renamed job beside a stale argument is a green run
    against no row.

Nothing here is a hand-written list of legs or of checks. The names are read from the first file,
the legs from the matrix the workflow carries, and the leg names are rendered by
`tools/gen_platform_table.py` - the same renderer that writes README's supported-platform table,
because three renderings of one name is how this list's consumers drift apart. The checkers are
read off the tree rather than named here: a `tools/` script is a check when its name is
`check_*.py` or when its own argument parser declares one of the flags a run of it is meant to
exit non-zero under, and a step carries it when its command invokes it with one of those flags. The suite run by `ctest` is
in scope by the same argument: it is a check this tree relies on, and it is the one every leg that
builds a binary is built around.

The scope is deliberately wider than the set of checks anyone has in mind: a checker added to
`tools/` is in scope the moment it lands, and the run that would go quiet is the run this exists
to keep from going quiet.

Usage:
  python3 tools/check_required_checks_are_armed.py
  python3 tools/check_required_checks_are_armed.py --required <file> --workflow <file>
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import re
import sys

import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent
REQUIRED = REPO / ".github" / "required-checks.txt"
WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"
TOOLS = REPO / "tools"
PLATFORM_TOOL = pathlib.Path(__file__).resolve().parent / "gen_platform_table.py"

# The runner this tree's own suite is run with. The suite is a check like the scripts below, and it
# is the one check every leg that builds a binary is built around.
SUITE = "ctest"

# The flags a `tools/` script declares when a run of it is meant to exit non-zero on a difference.
# Read as a declaration in the script's own source rather than assumed: a script that declares
# none is carried by any invocation of it.
VETO_FLAGS = ("--check", "--strict")

# A step carries a checker when its command invokes the script by its own path in this tree.
TOOL = re.compile(r"tools/([A-Za-z0-9_]+\.py)")

# The `if:` clauses this reader understands. A matrix clause restricts a step to the entries it
# holds of; the four status functions below say when a step is allowed to run and never which
# entries a job is drawn with, so none of them narrows the set of legs. Anything else stops the
# check rather than being read as a yes: a step this reader cannot evaluate is a leg it would
# claim coverage for without knowing.
CLAUSE = re.compile(r"^matrix\.([A-Za-z0-9_]+)\s*==\s*'([^']*)'$")
STATUS_FUNCTION = re.compile(r"^(always|success|failure|cancelled)\(\)$")

LEG_ARG = re.compile(r'--leg\s+"([^"]*)"')


def platform_tool():
    """The leg-name renderer `gen_platform_table.py` already owns, loaded from its path.

    Imported rather than copied: the names this check reasons about are the names that tool prints
    into README.md and that the branch-protection rule requires, and two renderings of one name is
    how the three drift apart.
    """
    spec = importlib.util.spec_from_file_location("_platform_table", PLATFORM_TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def required_names(path: pathlib.Path) -> list[str]:
    """The check names branch protection requires, comments stripped."""
    names = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()

        if line:
            names.append(line)

    return names


def runs_on(step: dict, entry: dict) -> bool:
    """Whether a step's `if:` lets it run on one matrix entry."""
    guard = step.get("if")

    if guard is None:
        return True

    text = str(guard).strip()

    if text.startswith("${{") and text.endswith("}}"):
        text = text[3:-2].strip()

    for raw_clause in text.split("&&"):
        clause = raw_clause.strip()

        if STATUS_FUNCTION.match(clause):
            continue

        match = CLAUSE.match(clause)

        if match is None:
            raise SystemExit(
                f"check_required_checks_are_armed: cannot evaluate the if: expression "
                f"{guard!r} in step {step.get('name')!r} - teach this script about it"
            )

        key, value = match.group(1), match.group(2)

        if str(entry.get(key)) != value:
            return False

    return True


def legs(workflow: dict, platform) -> list[tuple[str, str, dict]]:
    """Every CI leg as (rendered check name, job id, the matrix entry it is drawn from)."""
    rows = []

    for job_id, job in workflow["jobs"].items():
        template = job.get("name", job_id)

        for entry in platform.matrix_entries(job):
            rows.append((platform.expand(template, entry), job_id, entry))

    return rows


def steps_of(workflow: dict, platform) -> list[tuple[str, str, str, dict]]:
    """Every step as (step name, job id, leg, step), one row per leg it runs on.

    A step carrying an `if:` becomes one row per entry it passes on, so a step scoped to one leg of
    a matrix is that one leg here and not the job it sits in.
    """
    rows = []

    for job_id, job in workflow["jobs"].items():
        template = job.get("name", job_id)

        for entry in platform.matrix_entries(job):
            leg = platform.expand(template, entry)

            for step in job.get("steps", []):
                if runs_on(step, entry):
                    rows.append((step.get("name", ""), job_id, leg, step))

    return rows


def command_of(step: dict) -> str:
    """A step's command with the comment lines of its run block dropped.

    A commented-out invocation is not an invocation, and reading one as a check that still runs is
    the direction of error this whole check exists to refuse.
    """
    lines = str(step.get("run", "")).splitlines()

    return "\n".join(line for line in lines if not line.strip().startswith("#"))


def checkers(tools: pathlib.Path) -> dict[str, tuple[str, ...]]:
    """Every check this tree carries, as script name -> the veto flags its own source declares."""
    found = {}

    for script in sorted(tools.glob("*.py")):
        source = script.read_text("utf-8")
        declared = tuple(
            flag
            for flag in VETO_FLAGS
            if re.search(rf"add_argument\(\s*[\"']{re.escape(flag)}[\"']", source)
        )

        if script.name.startswith("check_") or declared:
            found[script.name] = declared

    return found


def carried(
    steps: list[tuple[str, str, str, dict]], name: str, flags: tuple[str, ...]
) -> list[tuple[str, str, str]]:
    """The (step name, leg, command) rows whose command runs `name` in one of its checking modes."""
    rows = []

    for step_name, _, leg, step in steps:
        command = command_of(step)

        if name not in TOOL.findall(command):
            continue

        if flags and not any(flag in command for flag in flags):
            continue

        rows.append((step_name, leg, command))

    return rows


def suite_rows(steps: list[tuple[str, str, str, dict]]) -> list[tuple[str, str, str]]:
    """The rows whose command runs the test suite."""
    return [
        (step_name, leg, command_of(step))
        for step_name, _, leg, step in steps
        if re.search(rf"\b{SUITE}\b", command_of(step))
    ]


def check_leg_arguments(workflow: dict, platform) -> list[str]:
    """Every `--leg` argument, against the names its own job renders to.

    The reading it feeds does not fail on a name that matches no row, so this is the only place the
    two can be held to each other: a job renamed in one place and not the other leaves the
    comparison in docs/build-facts.md running against nothing and reporting green.
    """
    failures = []

    for job_id, job in workflow["jobs"].items():
        template = job.get("name", job_id)

        for entry in platform.matrix_entries(job):
            names = [platform.expand(template, entry)]

            for step in job.get("steps", []):
                if not runs_on(step, entry):
                    continue

                for raw in LEG_ARG.findall(command_of(step)):
                    rendered = platform.expand(raw, entry)

                    if rendered not in names:
                        failures.append(
                            f"the step {step.get('name')!r} in job {job_id!r} passes "
                            f"--leg {rendered!r}, and that job's own name renders to {names} - the "
                            f"comparison runs against no row and reports green"
                        )

    return failures


def listing(rows: list[tuple[str, str, str]], required_legs: set[str]) -> str:
    """`n` leg(s), with a couple of names, for a one-line report of where a check runs."""
    legs_here = sorted({leg for _, leg, _ in rows})
    armed = [leg for leg in legs_here if leg in required_legs]
    shown = ", ".join(legs_here[:2]) + (f", +{len(legs_here) - 2} more" if len(legs_here) > 2 else "")

    return f"{len(legs_here)} leg(s): {shown}; {len(armed)} of them required"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="accepted for the invocation the workflow uses uniformly; this tool has no other "
        "mode, so a finding exits non-zero with or without it",
    )
    parser.add_argument("--required", type=pathlib.Path, default=REQUIRED)
    parser.add_argument("--workflow", type=pathlib.Path, default=WORKFLOW)
    args = parser.parse_args()

    if not args.required.is_file() or not args.workflow.is_file():
        print(
            f"check_required_checks_are_armed: missing {args.required} or {args.workflow}",
            file=sys.stderr,
        )
        return 1

    platform = platform_tool()
    workflow = yaml.safe_load(args.workflow.read_text(encoding="utf-8"))
    required = required_names(args.required)
    rows = legs(workflow, platform)
    steps = steps_of(workflow, platform)
    checks = checkers(TOOLS)

    if not required or not rows or not checks:
        print(
            f"check_required_checks_are_armed: read {len(required)} required name(s), {len(rows)} "
            f"leg(s) and {len(checks)} check(s). A check that can pass by reading nothing is not a "
            f"check",
            file=sys.stderr,
        )
        return 1

    all_legs = {name: (job_id, entry) for name, job_id, entry in rows}
    required_legs = {name for name in required if name in all_legs}
    failures: list[str] = []

    # --- The names ---------------------------------------------------------------------------
    print(f"check_required_checks_are_armed: {args.required} against {args.workflow}")
    print(f"\n  required names: {len(required)} - the workflow defines {len(rows)} leg(s)\n")

    for name in required:
        if name not in all_legs:
            failures.append(
                f"{args.required.name} names {name!r}, which is no leg of {args.workflow.name} - a "
                f"name that matches nothing is silently unenforced"
            )
            print(f"    {name:<34} NO LEG RENDERS THIS NAME")
            continue

        job_id, entry = all_legs[name]
        # Only the keys that job's name template reads: the rest of the entry is the leg's own
        # business and would bury the answer to "which leg renders this name".
        keys = re.findall(r"matrix\.([A-Za-z0-9_]+)", str(workflow["jobs"][job_id].get("name", "")))
        drawn = ", ".join(f"{key}={entry.get(key)}" for key in keys) or "single leg, no matrix"
        print(f"    {name:<34} <- job {job_id!r} ({drawn})")

    # --- The checks --------------------------------------------------------------------------
    print(f"\n  checks this tree carries: {len(checks)} under {TOOLS.name}/, plus the suite\n")

    instruments: list[tuple[str, tuple[str, ...], list[tuple[str, str, str]]]] = [
        (SUITE, (), suite_rows(steps))
    ]
    instruments += [
        (name, flags, carried(steps, name, flags)) for name, flags in checks.items()
    ]

    for name, flags, rows_here in instruments:
        if name == SUITE:
            label, detail = f"{SUITE} (the test suite)", f"the runner, not a script under {TOOLS.name}/"
        else:
            label, detail = f"{TOOLS.name}/{name}", f"declares {' '.join(flags) or 'no veto flag'}"
        steps_here = sorted({step for step, _, _ in rows_here})
        legs_here = sorted({leg for _, leg, _ in rows_here})
        armed = sorted(leg for leg in legs_here if leg in required_legs)

        print(f"    {label}\n        {detail}")

        if not rows_here:
            failures.append(
                f"{label} is a check this tree carries, and no step of {args.workflow.name} "
                f"invokes it - a check nothing runs stops no merge"
            )
            print("        INVOKED BY NO STEP")
        elif not armed:
            failures.append(
                f"{label} is run by {steps_here}, which runs on no leg {args.required.name} "
                f"requires - it reports to a reader and gates no merge. Its legs: {legs_here}"
            )
            print(f"        run by {steps_here}\n        ON NO REQUIRED LEG: {legs_here}")
        else:
            print(f"        run by {steps_here}")
            print(f"        {listing(rows_here, required_legs)}")

    # --- The leg arguments -------------------------------------------------------------------
    print()
    for failure in check_leg_arguments(workflow, platform):
        failures.append(failure)
        print(f"    {failure}")

    # --- The verdict -------------------------------------------------------------------------
    print()

    if failures:
        print(f"check_required_checks_are_armed: {len(failures)} finding(s)", file=sys.stderr)

        for failure in failures:
            print(f"  {failure}", file=sys.stderr)

        return 1

    print(
        f"check_required_checks_are_armed: every one of the {len(required)} required name(s) "
        f"renders to a leg, and every one of the {len(checks) + 1} check(s) this tree carries runs "
        f"on at least one of them"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
