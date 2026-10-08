#!/usr/bin/env python3
r"""Run every check the CI workflow arms on its full leg, from one command.

A contributor who touches a coefficient table, a bound in a document, a
device option row or the build-defaults seam owes a specific check a run, and
until now the only record of which is ``.github/workflows/ci.yml`` and the
"Build and test" section of ``CONTRIBUTING.md``. A list written here would be
a third rendering of that set and would drift from both. So the checks are
read off the workflow at run time: this script takes the steps of one leg -
the one whose matrix entry is ``compiler == 'gcc' and build_type ==
'Release'``, which is the leg the source-integrity gates are scoped to - and
runs every ``tools/*.py`` invocation those steps carry, in the workflow's own
order, with the workflow's own arguments.

What it is not. It is not a gate: it runs the checks and reports their exit
codes, and the checks themselves remain the authority on what they read. It
runs no step of the leg that is not a checker under ``tools/`` - the configure
and the build, ``ctest``, the boundary standalone, the Doxygen gate and the
two extra build-default trees are named in the output as steps this command
does not cover, because a runner that stayed silent about them would read as
though it had run everything. It writes no file, stages nothing and commits
nothing.

A check that could not run is reported as ``skipped`` with the reason, never
as one that passed, and the summary counts it. Two things are read as a
missing input rather than a failure to fix here, because the tool that needs
them refuses on its own terms and would refuse after its expensive work
rather than before it:

  * the formatter. ``tools/gen_boys_coefficients.py --check`` compares the
    committed header byte for byte against a freshly formatted one, and CI
    installs clang-format 18.1.3 into ``.venv`` for it. The same three
    candidates that tool's own ``find_clang_format`` reads are read here, in
    the same order, so that a machine which has none is told so in a second
    rather than after the derivation the comparison is made of.
  * mpmath, for the generators that import it. The interpreter is the
    checkout's ``.venv`` where there is one and the running interpreter
    otherwise, which is the substitution ``.venv/bin/python`` needs on a
    machine whose modules are installed globally.

``--fast`` runs the checks that read the tree and skips the ones that pay for
a derivation or a compile, each named in the output with the reason it was
skipped. The slow set is not a guess about cost: every member of it either
re-derives the committed tables at full precision or compiles a driver against
the revision's own sources, and the reason printed for each says which.

Usage:
  python tools/check_all.py                 # every check this leg arms
  python tools/check_all.py --fast          # the tree-reading checks only
  python tools/check_all.py --leg "linux-x86 clang Debug"
  python tools/check_all.py --check         # the same run; exits non-zero on failure
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import NamedTuple

import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent
CI_YML = REPO / ".github" / "workflows" / "ci.yml"
TOOLS = REPO / "tools"
PLATFORM_TOOL = pathlib.Path(__file__).resolve().parent / "gen_platform_table.py"

# The matrix entry of the leg whose steps carry the source-integrity gates. It is the one leg the
# gates that read no build output are scoped to, so it is the leg a contributor's checks are on.
LEG_KEYS = (("compiler", "gcc"), ("build_type", "Release"))

# The `if:` clauses this reader understands, as in tools/check_required_checks_are_armed.py: a
# matrix clause restricts a step to the entries it holds of, and the four status functions say when
# a step may run and never which entries a job is drawn with. Anything else stops this script
# rather than being read as a yes.
CLAUSE = re.compile(r"^matrix\.([A-Za-z0-9_]+)\s*==\s*'([^']*)'$")
STATUS_FUNCTION = re.compile(r"^(always|success|failure|cancelled)\(\)$")

# A step carries a check when its command invokes a script under tools/ by its own path.
TOOL = re.compile(r"tools/([A-Za-z0-9_]+\.py)")

# The expression a step's `env:` may carry and the ones this reader can resolve.
GH_EXPRESSION = re.compile(r"\$\{\{\s*([A-Za-z0-9_.]+)\s*\}\}")

# The checks a `--fast` run skips, each with the reason it is slow. Both members pay for work that
# is not a reading of the tree: the two generators re-derive what they compare against, and the two
# compile-based checks build a driver against the revision's own sources. The words in each reason
# are the tree's own, and the entry is held to the tree: a key that names no check this leg arms is
# reported rather than ignored, because a stale entry here is a check the fast run would run anyway
# or one it would skip for a reason nobody can read.
SLOW = {
    "gen_boys_coefficients.py": (
        "re-derives the coefficient header and the reference grid at full precision before it "
        "compares anything; ci.yml calls it \"the longest step anywhere in this matrix\""
    ),
    "gen_boys_accuracy_gate_reference.py": (
        "a full render is 56694 mpmath evaluations at 80 dps, measured in ci.yml at "
        "\"2m20s to 2m40s end to end\""
    ),
    "check_class_combinations.py": (
        "compiles one instantiation of each one-order shape at the orders packing axis, and "
        "requires the compile to fail; ci.yml: \"the check proves by compiling rather than by "
        "quoting\""
    ),
    "check_combination_bounds.py": (
        "compiles one driver against the revision's own sources and runs it once; its own "
        "docstring: \"The compile is the slow part\""
    ),
}

# One reading of the chosen interpreter's imports per run, rather than one per check that asks.
_MPMATH: dict[str, bool] = {}


class Check(NamedTuple):
    """One checker invocation: what the workflow calls it, and the argument vector it runs."""

    step: str
    argv: tuple[str, ...]
    env: dict[str, str]

    @property
    def tool(self) -> str:
        return pathlib.Path(self.argv[0]).name

    @property
    def command(self) -> str:
        return " ".join(self.argv)


def platform_tool():
    """`gen_platform_table.py`, loaded from its path rather than copied.

    It is the tree's renderer for a leg's check name - README's supported-platform table and
    check_required_checks_are_armed.py both read the name through it - and this script's default
    leg is selected by that name, so a second rendering of it is how a contributor's run and CI
    would come to disagree about which leg carries what.
    """
    spec = importlib.util.spec_from_file_location("_platform_table", PLATFORM_TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def step_runs_on(step: dict, entry: dict) -> bool:
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
                f"check_all: cannot evaluate the if: expression {guard!r} in step "
                f"{step.get('name')!r} - teach this script about it"
            )

        key, value = match.group(1), match.group(2)

        if str(entry.get(key)) != value:
            return False

    return True


def command_lines(step: dict) -> list[str]:
    """A step's `run:` block as command lines, continuations joined and comments dropped.

    A commented-out invocation is not an invocation, and reading one as a check that still runs is
    the direction of error a runner must not make.
    """
    lines: list[str] = []
    pending = ""

    for raw in str(step.get("run", "")).splitlines():
        line = raw.strip()

        if not pending and line.startswith("#"):
            continue

        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue

        joined = (pending + line).strip()
        pending = ""

        if joined:
            lines.append(joined)

    if pending.strip():
        lines.append(pending.strip())

    return lines


def step_environment(step: dict, entry: dict) -> dict[str, str]:
    """A step's `env:` with the expressions this reader can resolve, for the checks it runs."""
    resolved: dict[str, str] = {}

    for key, value in (step.get("env") or {}).items():
        text = str(value)

        def substitute(match: re.Match, text: str = text) -> str:
            name = match.group(1)

            if name == "github.workspace":
                return str(REPO)

            if name.startswith("matrix."):
                return str(entry.get(name[len("matrix."):], match.group(0)))

            return match.group(0)

        resolved[str(key)] = GH_EXPRESSION.sub(substitute, text)

    return resolved


def invocations(step: dict, entry: dict, strict: bool) -> list[Check]:
    """Every checker invocation a step carries, as the workflow spells it.

    `strict` is for the leg this run is about: a command there that names a script under tools/
    without spelling it as the bare relative path a `run:` block in this workflow otherwise uses -
    the option-plan job writes one inside a `$(...)` substitution - is a spelling this reader
    cannot turn into an argument vector, and reading it as one it can is how a check would be run
    with the wrong arguments. Elsewhere it is reported and skipped: the leg scan is a statement
    about where the other checks live, not a run of them.
    """
    found: list[Check] = []
    env = step_environment(step, entry)

    for line in command_lines(step):
        try:
            tokens = shlex.split(line, posix=True)
        except ValueError:
            # A line this reader cannot tokenize - a shell construct whose quoting it does not
            # model - is not one it may hand a checker's arguments to.
            if not strict:
                continue

            raise SystemExit(
                f"check_all: the step {step.get('name')!r} carries the line {line!r}, which "
                f"this script cannot split into an argument vector - teach it about this spelling"
            ) from None

        start = next(
            (index for index, token in enumerate(tokens) if TOOL.search(token)),
            None,
        )

        if start is None:
            continue

        argv = tuple(tokens[start:])

        if not TOOL.fullmatch(argv[0]):
            if not strict:
                continue

            raise SystemExit(
                f"check_all: the step {step.get('name')!r} invokes {argv[0]!r}, which this "
                f"script cannot resolve to a script under tools/ - teach it about this spelling"
            )

        found.append(Check(str(step.get("name", "")), argv, env))

    return found


def legs(workflow: dict, platform) -> list[tuple[str, str, dict]]:
    """Every leg of the workflow as (rendered check name, job id, matrix entry)."""
    rows = []

    for job_id, job in workflow["jobs"].items():
        template = job.get("name", job_id)

        for entry in platform.matrix_entries(job):
            rows.append((platform.expand(template, entry), job_id, entry))

    return rows


def select_leg(workflow: dict, platform, wanted: str | None):
    """The leg to read the checks off: named by `--leg`, or the gcc/Release one by its matrix entry."""
    rows = legs(workflow, platform)

    if wanted is not None:
        matches = [row for row in rows if row[0] == wanted]

        if not matches:
            known = "\n  ".join(sorted(name for name, _, _ in rows))
            raise SystemExit(f"check_all: no leg is named {wanted!r}. The legs are:\n  {known}")

        return matches[0]

    matches = [
        row
        for row in rows
        if all(str(row[2].get(key)) == value for key, value in LEG_KEYS)
    ]

    if len(matches) != 1:
        named = ", ".join(repr(name) for name, _, _ in matches) or "none"
        raise SystemExit(
            f"check_all: {len(matches)} leg(s) are drawn with "
            f"{' and '.join(f'{key} == {value!r}' for key, value in LEG_KEYS)} ({named}), and this "
            f"script reads the checks off exactly one. Name the leg with --leg"
        )

    return matches[0]


def collect(workflow: dict, platform, job_id: str, entry: dict) -> tuple[list[Check], list[str]]:
    """The leg's checker invocations in workflow order, and the steps of it that carry none."""
    checks: list[Check] = []
    others: list[str] = []

    for step in workflow["jobs"][job_id].get("steps", []):
        if not step_runs_on(step, entry):
            continue

        found = invocations(step, entry, strict=True)

        if found:
            checks.extend(found)
        elif "run" in step:
            # A step that runs something and is not a checker under tools/. The ones that run
            # nothing - the checkout, the toolkit action - are not steps this run could cover or
            # leave out, and naming them would bury the ones it does.
            others.append(str(step.get("name", "")) or "(a step with no name)")

    return checks, others


def invoked_tools(workflow: dict, platform, only: str | None = None) -> dict[str, list[str]]:
    """Every checker the workflow invokes, as tool -> the legs whose steps invoke it."""
    found: dict[str, list[str]] = {}

    for leg, job_id, entry in legs(workflow, platform):
        if only is not None and leg != only:
            continue

        for step in workflow["jobs"][job_id].get("steps", []):
            if not step_runs_on(step, entry):
                continue

            for check in invocations(step, entry, strict=False):
                found.setdefault(check.tool, [])
                if leg not in found[check.tool]:
                    found[check.tool].append(leg)

    return found


def armed_elsewhere(workflow: dict, platform, here: str) -> list[tuple[str, list[str]]]:
    """The checkers this workflow arms on another leg, as (tool, legs), for the runner to name.

    A step of the leg above is what a contributor's checks are, and a check that lives on another
    leg - the example transcripts on the Debug leg, say - is not run here. Saying so is the
    difference between a runner that covered the workflow and one that reads as though it had.
    """
    on_this_leg = set(invoked_tools(workflow, platform, only=here))

    return sorted(
        (tool, legs_here)
        for tool, legs_here in invoked_tools(workflow, platform).items()
        if tool not in on_this_leg
    )


def interpreter() -> tuple[str, str]:
    """The interpreter to run the checks with, and how it was chosen."""
    for candidate in (REPO / ".venv" / "bin" / "python", REPO / ".venv" / "Scripts" / "python.exe"):
        if candidate.is_file():
            return str(candidate), f"this checkout's {candidate.relative_to(REPO)}"

    return sys.executable, "the interpreter running this script (this checkout has no .venv)"


def formatter() -> str | None:
    """clang-format, by the three candidates the coefficient generator's own finder reads.

    Read in the same order, $CLANG_FORMAT first: the generator compares the committed header
    against a freshly formatted one, CI names the pinned wheel through that variable, and the
    generator refuses rather than skipping when it finds none.
    """
    candidates = [
        os.environ.get("CLANG_FORMAT"),
        shutil.which("clang-format"),
        r"C:\Program Files\Microsoft Visual Studio\18\Community"
        r"\VC\Tools\Llvm\x64\bin\clang-format.exe",
    ]

    for candidate in candidates:
        if candidate and pathlib.Path(candidate).exists():
            return str(candidate)

    return None


def imports_mpmath(tool: str) -> bool:
    """Whether a generator imports mpmath, read from the tool's own source."""
    try:
        source = (TOOLS / tool).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False

    return bool(re.search(r"^\s*(import mpmath|from mpmath import)", source, re.MULTILINE))


def mpmath_available(runner: str) -> bool:
    """Whether the interpreter this run uses can import mpmath."""
    if "mpmath" not in _MPMATH:
        _MPMATH["mpmath"] = (
            subprocess.run(
                [runner, "-c", "import mpmath"], capture_output=True, check=False
            ).returncode
            == 0
        )

    return _MPMATH["mpmath"]


def skip_reason(check: Check, runner: str, have_formatter: bool) -> str | None:
    """Why a check cannot run here, or None.

    Two inputs, both of them things the workflow installs and a development checkout may not have.
    Neither is a reason to report a pass: the check is named as skipped, with the remedy, and the
    summary counts it.
    """
    if not have_formatter and "CLANG_FORMAT" in check.env:
        return (
            "needs clang-format, and this machine has none - not at $CLANG_FORMAT, not on PATH, "
            "not in the Visual Studio install. CI installs clang-format==18.1.3 into .venv for "
            "it, and it compares the committed header against a freshly formatted one"
        )

    if imports_mpmath(check.tool) and not mpmath_available(runner):
        return "its generator imports mpmath, which this interpreter does not have (CI pins 1.4.1)"

    return None


def run(check: Check, runner: str, timeout: float) -> tuple[str, int, str]:
    """Run one check, returning (what happened, exit code, output)."""
    env = dict(os.environ)
    env.update(check.env)
    started = time.monotonic()

    try:
        finished = subprocess.run(
            [runner, *check.argv],
            cwd=str(REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout or None,
            check=False,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - started
        return f"killed at {elapsed:.0f}s by --timeout", 124, ""

    return f"{time.monotonic() - started:.1f}s", finished.returncode, (
        (finished.stdout or "") + (finished.stderr or "")
    )


def tail(output: str, lines: int = 8) -> list[str]:
    """The last few non-empty lines of a check's output, which is where its reason is."""
    kept = [line for line in output.splitlines() if line.strip()]
    return kept[-lines:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if any check fails; the default run behaves the same way, so this "
        "flag is accepted for the invocation the workflow uses uniformly",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="skip the checks that pay for a derivation or a compile, naming each one skipped "
        "and the reason it is slow; the checks that read the tree are all that run",
    )
    parser.add_argument(
        "--leg",
        default=None,
        help="the CI leg whose steps are read for the checks (default: the compiler=gcc, "
        "build_type=Release entry of the workflow's matrix)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="seconds after which a single check is killed and reported as a timeout, not as a "
        "pass (default 0: no limit, which is what CI does per step with timeout-minutes)",
    )
    args = parser.parse_args()

    if not CI_YML.is_file():
        print(f"check_all: {CI_YML} is missing", file=sys.stderr)
        return 1

    platform = platform_tool()
    workflow = yaml.safe_load(CI_YML.read_text(encoding="utf-8"))
    leg, job_id, entry = select_leg(workflow, platform, args.leg)
    checks, other_steps = collect(workflow, platform, job_id, entry)

    if not checks:
        print(
            f"check_all: the leg {leg!r} carries no checker under {TOOLS.name}/, and a run that "
            f"checked nothing is not one that passed",
            file=sys.stderr,
        )
        return 1

    runner, how = interpreter()
    found = formatter()
    have_formatter = found is not None

    print(f"check_all: the checks {CI_YML.relative_to(REPO)} arms on the leg {leg!r}")
    print(f"check_all: {len(checks)} checker invocation(s), run with {runner}")
    print(f"check_all: {how}")
    print(f"check_all: clang-format: {found or 'not found on this machine'}")

    if args.fast:
        print("check_all: --fast - the checks that derive or compile are skipped, with reasons")
    else:
        # Named before the run rather than after it: the slowest of these is a derivation that
        # ci.yml calls the longest step in the matrix, and a contributor who wanted the fast
        # answer should learn that before paying for it rather than after.
        slow = [check for check in checks if check.tool in SLOW and not skip_reason(
            check, runner, have_formatter)]

        if slow:
            print(f"check_all: {len(slow)} check(s) in this run are slow; --fast skips them:")

            for check in slow:
                print(f"         {check.tool}: {SLOW[check.tool]}")

    print()

    failures: list[tuple[Check, str]] = []
    skipped: list[tuple[Check, str]] = []

    for index, check in enumerate(checks, start=1):
        reason = skip_reason(check, runner, have_formatter)

        if reason is None and args.fast and check.tool in SLOW:
            reason = f"--fast: {SLOW[check.tool]}"

        if reason is not None:
            skipped.append((check, reason))
            print(f"  [{index:>2}] skipped        {check.command}  # {check.step}")
            print(f"       {reason}")
            continue

        # The formatter the workflow pins into .venv, resolved to the one this machine has, so
        # that a check whose step names a path that does not exist here still runs against a
        # formatter rather than against nothing.
        if found and "CLANG_FORMAT" in check.env:
            check.env["CLANG_FORMAT"] = found

        verdict, code, output = run(check, runner, args.timeout)
        print(f"  [{index:>2}] exit {code:<3} {verdict:<22} {check.command}  # {check.step}")

        if code != 0:
            failures.append((check, output))

            for line in tail(output):
                print(f"       | {line}")

    # A SLOW entry is stale when the tool it names is a checker of no leg at all: an entry for a
    # check this leg does not carry is ordinary, because the other legs carry different steps.
    stale = sorted(
        tool for tool in SLOW if tool not in invoked_tools(workflow, platform)
    )

    print()
    print(f"check_all: {len(checks) - len(skipped)} of {len(checks)} check(s) ran, "
          f"{len(failures)} failed, {len(skipped)} skipped")

    for tool in stale:
        print(
            f"check_all: SLOW names {TOOLS.name}/{tool}, which is no check of the leg {leg!r} - "
            f"the entry outlives the check it was written for, and the fast run would skip a "
            f"check by a name it no longer runs under"
        )

    if other_steps:
        print()
        print("check_all: steps of this leg that are not checkers here, and so are not run:")
        print(f"  {', '.join(other_steps)}")

    elsewhere = armed_elsewhere(workflow, platform, leg)

    if elsewhere:
        print()
        print(f"check_all: {len(elsewhere)} checker(s) this workflow arms on another leg, and "
              f"not on this one:")
        for tool, legs_here in elsewhere:
            shown = ", ".join(legs_here[:2]) + (
                f", +{len(legs_here) - 2} more" if len(legs_here) > 2 else ""
            )
            print(f"  {TOOLS.name}/{tool}: {shown}")

    if failures:
        print()
        print(f"check_all: {len(failures)} of {len(checks)} check(s) failed", file=sys.stderr)

        for check, _ in failures:
            print(f"  {check.step}: {check.command}", file=sys.stderr)

        return 1

    if stale:
        return 1

    print()
    print(
        f"check_all: every check the leg {leg!r} arms that could run here passed, and the "
        f"{len(skipped)} that could not are named above with their reason"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
