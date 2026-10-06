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
`check_*.py` or when its own argument parser declares a flag whose own words say a run of it exits
non-zero, and a step carries it when its command invokes it with one of those flags. The spelling
of that flag is the tree's business and not this file's: most of the tools say it of `--check` and
one of `--strict`, which is why a declaration spelling its checking mode either of those two is
read as the check it is without saying anything itself, and a tool that spells it otherwise is
read on the same terms - the report prints the spellings it read and how many declarations state
each. The suite run by `ctest` is in scope by the same argument: it is a check
this tree relies on, and it is the one every leg that builds a binary is built around.

The scope is deliberately wider than the set of checks anyone has in mind: a checker added to
`tools/` is in scope the moment it lands, and the run that would go quiet is the run this exists
to keep from going quiet.

What the reading cannot see, said here rather than left to be found by the run that goes quiet. It
reads a tool's own declarations, so a checking mode the declaration does not state - a flag whose
help names neither a non-zero exit nor a failure - is not read as one, and a tool that takes that
form and is not named `check_*.py` is no check to this file. Two nets narrow that: a `check_*.py`
that declares a mode this reader cannot name is a finding rather than a check carried by any
invocation of it, and every script the reading puts in no check at all is named in the report,
with the steps that invoke it, on every run.

THE CHECKS NO LEG ARMS. A check whose input no leg can produce is not a check to wire, and the
alternative to saying so is a step that runs it to no purpose. Those checks are held in `UNARMED`
below, each with the reason it is there and what would arm it, so the reason a check is not armed
is stated rather than left to be read off a silence. The list is not a place to park a red check,
and it is checked against that: an entry fails this run when it names no check this reading
carries, when it states no reason, and when a step does invoke the check after all - a list that
can go quiet on a wired check is a licence and not a record. Every entry is printed on every run.

An entry may name a check this reading could not classify as well as one it carries. The harm
behind the unclassified verdict is that every invocation of the check reads as carried, and no
invocation of a check nothing runs is - so the entry states the reading and the check's own mode
question is not asked of it. What is not given up is the net: a `check_*.py` whose mode this
reader cannot name and which no entry accounts for is still a finding.

Usage:
  python3 tools/check_required_checks_are_armed.py
  python3 tools/check_required_checks_are_armed.py --required <file> --workflow <file>
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import pathlib
import re
import sys
from typing import NamedTuple

import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent
REQUIRED = REPO / ".github" / "required-checks.txt"
WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"
TOOLS = REPO / "tools"
PLATFORM_TOOL = pathlib.Path(__file__).resolve().parent / "gen_platform_table.py"

# The runner this tree's own suite is run with. The suite is a check like the scripts below, and it
# is the one check every leg that builds a binary is built around.
SUITE = "ctest"

# The words a tool's own declaration of a flag uses when a run with it is meant to exit non-zero
# on a difference. `--check` and `--strict` are the spellings this tree settled on, and they are
# spellings rather than a list of tools: nine declarations say "exit non-zero"/"exits nonzero" of
# the one and one says "also fail on multiplier-unswept findings" of the other, so both are read
# off the tree by the rule below. A tool whose checking flag is named something else enters the
# vocabulary on the same terms, by saying the same thing of it.
FAILING_OUTCOME = re.compile(r"non-?zero|\bfail(?:s|ed|ure)?\b", re.IGNORECASE)

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

# The checks this tree carries and no leg of the workflow arms: file name -> why, and what would
# arm it. Every reason is a fact about the check's INPUT - a report a leg cannot produce, a record
# a leg cannot re-make, a build of the check's own - and never its cost, because cost is work and a
# check that is merely slow is a check to arm and measure. `unarmed_reasons` holds the list to
# those terms on every run.
# Every check that once stood here is armed: `check_class_combinations` is pointed at a
# committed report, `check_combination_bounds` builds its driver on a leg, and
# `check_gate_covers_combinations` has a step of its own. The entry for the last one was
# put back by hand and this tool refused it - "UNARMED names a check the step(s) do
# invoke" - which is the behaviour that makes the list worth keeping empty rather than
# stale. A check that cannot be armed yet belongs here with its reason; none now does.
UNARMED = {}


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


class Declaration(NamedTuple):
    """One `add_argument` call: the flag it declares, whether it is a mode, and its own words."""

    flag: str
    switch: bool
    text: str


class Inventory(NamedTuple):
    """What a reading of `tools/` concludes about which of its scripts are checks.

    `checks` is what the rest of this file reasons about: script name -> the flags a run of it is
    meant to exit non-zero under. `words` is where those spellings were read from, carried so the
    report prints the derivation rather than asking a reader to trust it. `unclear` is the scripts
    this reading cannot classify - named a check by their own name, declaring a mode, none of them
    a mode the tree states a failing outcome for - and each is a finding, because the alternative
    is a step whose mode nothing can check and every invocation of which reads as carried.
    `unreadable` is the scripts whose declarations could not be read at all. `not_checks` is the
    scripts this reading puts in no check: named in the report on every run, and never gated,
    because a tool that regenerates a file is not a defect.
    """

    checks: dict[str, tuple[str, ...]]
    words: dict[str, list[str]]
    unclear: dict[str, list[str]]
    unreadable: list[str]
    not_checks: list[str]


def declarations(source: str) -> list[Declaration]:
    """Every flag a script declares, read from its own argument parser.

    Read as a declaration in the script's own source rather than assumed, which is the point: the
    flag that fails a run is whatever the tool says it is, and a tool that declares no flag at all
    is a check carried by any invocation of it.
    """
    found: list[Declaration] = []

    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue

        if node.func.attr != "add_argument":
            continue

        flag = next(
            (
                argument.value
                for argument in node.args
                if isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
                and argument.value.startswith("-")
            ),
            None,
        )

        if flag is None:
            # A positional: a value the caller supplies, never a mode the run is put into.
            continue

        action = next(
            (
                str(keyword.value.value)
                for keyword in node.keywords
                if keyword.arg == "action"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ),
            "",
        )

        found.append(
            Declaration(
                flag,
                action in ("store_true", "store_false"),
                " ".join(
                    part.value
                    for part in ast.walk(node)
                    if isinstance(part, ast.Constant) and isinstance(part.value, str)
                ),
            )
        )

    return found


def vocabulary(declared: dict[str, list[Declaration]]) -> dict[str, list[str]]:
    """The spellings a run of a tool exits non-zero under, as flag -> the tools that say so.

    Read off the declarations themselves: a mode whose own words name a non-zero exit or a failure
    is a spelling of the checking form whatever it is called, and the `--check` and `--strict` of
    this tree are in the set because ten declarations say it of them rather than because they are
    written down anywhere. A value the caller supplies is not a mode and is not read as one, so a
    threshold whose help says a mismatch "fails the run" is not mistaken for the failing flag.
    """
    words: dict[str, list[str]] = {}

    for name, found in declared.items():
        for declaration in found:
            if declaration.switch and FAILING_OUTCOME.search(declaration.text):
                words.setdefault(declaration.flag, []).append(name)

    return words


def inventory(tools: pathlib.Path) -> Inventory:
    """Read `tools/` for the checks it carries, and for the scripts this reading says are none."""
    declared: dict[str, list[Declaration]] = {}
    unreadable: list[str] = []

    for script in sorted(tools.glob("*.py")):
        try:
            declared[script.name] = declarations(script.read_text("utf-8"))
        except SyntaxError:
            # A script this reader cannot parse is one whose declarations it cannot read, which is
            # the same blindness as a mode it cannot name. A finding, not a script passed over.
            declared[script.name] = []
            unreadable.append(script.name)

    words = vocabulary(declared)
    checks: dict[str, tuple[str, ...]] = {}
    unclear: dict[str, list[str]] = {}

    for name, found in declared.items():
        flags = tuple(sorted({item.flag for item in found if item.switch and item.flag in words}))

        if flags:
            checks[name] = flags
        elif name.startswith("check_") and name not in unreadable:
            modes = sorted({item.flag for item in found if item.switch})

            if modes:
                unclear[name] = modes
            else:
                # Declares nothing to be put into: every invocation of it is the checking one.
                checks[name] = ()

    return Inventory(
        checks,
        words,
        unclear,
        unreadable,
        sorted(set(declared) - set(checks) - set(unclear) - set(unreadable)),
    )


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


def unarmed_reasons(found: Inventory, steps: list[tuple[str, str, str, dict]]) -> list[str]:
    """How the `UNARMED` list can itself be wrong, one failure per way.

    Three ways, and none of them is a difference of opinion about the library. An entry naming no
    check this reading carries outlives the check it was written for, and the next check to take
    that name inherits an exemption nobody gave it. An entry stating no reason is a decision with
    its reason dropped, which is the shape a suppression list arrives in. And an entry whose check
    a step does invoke is a licence to leave the check unarmed that the workflow has already
    outgrown - so it is a finding rather than a quiet no-op.

    An entry may name a check this reading could not classify as well as one it carries. The
    verdict that makes an unclassified check a finding is that every invocation of it reads as a
    check carried, and a check no step invokes is carried by no invocation - so the ambiguity
    that verdict reports has no consequence for a check that is not armed, and the entry carries
    the reading instead of the finding.
    """
    failures: list[str] = []
    known = set(found.checks) | set(found.unclear)

    for name, reason in sorted(UNARMED.items()):
        if name not in known:
            failures.append(
                f"UNARMED names {name!r}, and this reading carries no check under that name - the "
                f"entry outlives the check it was written for, and a check that later takes the "
                f"name inherits an exemption nobody gave it"
            )

        if not reason.strip():
            failures.append(
                f"UNARMED names {name!r} and states no reason. A check left unarmed is a decision, "
                f"and a decision with its reason dropped is a suppression list"
            )

        invoked = sorted({step for step, _, _ in carried(steps, name, ())})

        if invoked:
            failures.append(
                f"UNARMED names {name!r} and the step(s) {invoked} do invoke it - a step was added "
                f"and the entry was not removed, so the list would go quiet on a check the "
                f"workflow arms"
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
    found = inventory(TOOLS)
    checks = found.checks

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

    # The vocabulary the reading above is made of, printed: a reader who disagrees with a verdict
    # sees the declarations it was read from rather than being asked to take it on trust.
    stated = ", ".join(
        f"{flag} (stated by {len(names)})" for flag, names in sorted(found.words.items())
    )
    print(f"    the mode a run exits non-zero under: {stated}")
    print("    read off the tools' own declarations, and spelled by no list here\n")

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

        if not rows_here and name in UNARMED:
            print(f"        NOT ARMED: {UNARMED[name]}")
        elif not rows_here:
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

    # --- The checks no leg arms, held to their own terms --------------------------------------
    # The reasons above are the tree's claim about why a check is not armed, so they are checked
    # here rather than trusted: an entry that has outlived its check, lost its reason, or been
    # overtaken by a step is the list failing rather than the list working.
    print()

    for failure in unarmed_reasons(found, steps):
        failures.append(failure)
        print(f"    {failure}")

    # --- What this reading will not call a check ----------------------------------------------
    # Two verdicts, and only the first is a gate. A script named a check by its own name whose
    # modes the reader cannot name is the case that used to go quiet, so it fails here. A script
    # this reading puts in no check at all is named and left alone: a generator is not a defect,
    # and the reader who brought one in needs to see it said rather than infer it from silence.
    print(f"\n  no check to this reading: {len(found.not_checks)} under {TOOLS.name}/\n")

    for name in found.not_checks:
        invoked = sorted({step for step, _, _ in carried(steps, name, ())})
        where = f"invoked by {invoked}" if invoked else "invoked by no step"
        print(f"    {TOOLS.name}/{name}: {where}")

    if found.unclear or found.unreadable:
        print()

    for name, modes in sorted(found.unclear.items()):
        if name in UNARMED:
            print(f"    {TOOLS.name}/{name}\n        NOT ARMED: {UNARMED[name]}")
            continue

        failures.append(
            f"{TOOLS.name}/{name} is named a check by its own name and declares {modes}, and none "
            f"of them says a run of it exits non-zero - this reader cannot say which invocation of "
            f"it fails, so every one of them reads as a check carried. State the failing mode in "
            f"the flag's own help, which is the form the rest of the tree states it in"
        )
        print(f"    {TOOLS.name}/{name}\n        NO MODE THIS READER CAN NAME: {modes}")

    for name in found.unreadable:
        failures.append(
            f"{TOOLS.name}/{name} could not be read: its source does not parse here, so nothing "
            f"can say which of its modes fails, and it is not read as a check"
        )
        print(f"    {TOOLS.name}/{name}\n        UNREADABLE")

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
        f"on at least one of them, or is one of the {len(UNARMED)} declared not armed with its "
        f"reason printed above"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
