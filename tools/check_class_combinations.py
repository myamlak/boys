#!/usr/bin/env python3
"""Every combination a class can be instantiated at is a combination the probe measured.

The option probe's report is read as an account of the library, so a class the report states with
one measured combination is read as a class that has one. That reading was false for thirteen of
the seventeen host classes: the probe enumerated its space once per precision LANE and measured
the whole of it through the all-orders entry, giving every other shape a single row called with no
policy argument. A class whose entry is policy-templated has as many combinations as the axes
offer, and the entry is instantiated at each of them or the compiler refuses it.

What a class offers is read from the library here and not written down:

  * the axis members come from the probe's own report, which reads them from the library's tables
    (``BoysEvalSchemes``, ``BoysFitGranularities``, ``BoysPackAxes``, ``BoysDivisionForms``,
    ``BoysRegionBExps`` and the lane's routes);
  * which shapes are one-order shapes comes from the library's own ``Shape`` enumerators — the
    documentation of each names the question it answers, and "one order" is a reading of that text
    rather than a list in this file;
  * that a one-order shape really cannot take the orders packing axis is not assumed: the check
    compiles one instantiation of each one-order shape at that axis value and requires it to FAIL,
    and one without it and requires it to PASS. A check that could not fail would be a decoration.

usage:
  check_class_combinations.py --report <report.txt> [--root DIR] [--no-compile]
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
import tempfile


class Reader:
    """The library read at one revision, or from a scratch tree already exported from one.

    A claim about what the library contains is a claim about a revision: the working tree can hold
    another writer's half-finished edit, and a reading of that state is wrong about the library in
    a way no reader of the output can see. Every header is read with ``git show <rev>:<path>`` and
    the revision is printed, and a compile is run in a tree exported from that revision.
    """

    def __init__(self, root: pathlib.Path, rev: str):
        self.root = root
        self.rev = rev
        self._text: dict[str, str] = {}
        self.exported: pathlib.Path | None = None

    def text(self, relative: str) -> str:
        if relative in self._text:
            return self._text[relative]

        if self.exported is not None:
            path = self.exported / relative
            value = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        else:
            done = subprocess.run(
                ["git", "-C", str(self.root), "show", f"{self.rev}:{relative}"],
                capture_output=True, text=True, encoding="utf-8", errors="replace")
            value = done.stdout if done.returncode == 0 else ""

        self._text[relative] = value
        return value

    def export(self) -> pathlib.Path:
        if self.exported is not None:
            return self.exported

        scratch = pathlib.Path(tempfile.mkdtemp(prefix="boys-check-"))
        archived = subprocess.run(["git", "-C", str(self.root), "archive", self.rev],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        unpacked = subprocess.run(["tar", "-x", "-C", str(scratch)], input=archived.stdout)

        if archived.returncode != 0 or unpacked.returncode != 0:
            raise RuntimeError(f"git archive {self.rev} did not export: {archived.stderr!r}")

        self.exported = scratch
        return scratch

    def rev_id(self) -> str:
        done = subprocess.run(["git", "-C", str(self.root), "rev-parse", self.rev],
                              capture_output=True, text=True)
        return done.stdout.strip() or self.rev


# The one line both host instruments print, spelled once so that the two agree by construction
# rather than by a reader noticing a difference. It is the reconciliation this check exists beside
# `tools/combination_matrix.py` for: the two answer the same question from the same report and must
# answer it with the same number.
HOST_TOTAL = ("host total — combination(s) a class can be instantiated at and this run did "
              "not measure: {total}")

# The packing axis member a one-order shape cannot be instantiated at: one order has one order to
# put in a vector lane, so the axis that fills a lane with a ladder's orders has nothing to fill it
# with. The library's own entry bodies assert this; the check compiles it rather than quoting it.
ORDERS_AXIS = "kOrders"

# The shapes whose entry evaluates one order, by the library's own documentation of each Shape
# enumerator in include/boys/boys.hpp. A shape whose doc names one order has no ladder, and the
# orders axis is not an axis on it.
# Each enumerator carries its documentation after `///<`. Only the first member of the enum
# carries `= 0`, so the initialiser is optional here.
SHAPE_ENUM = re.compile(r"^\s*(k[A-Za-z0-9]+)\s*(?:=\s*\d+)?\s*,?\s*///<\s*(.*)$", re.M)
ONE_ORDER = re.compile(r"\bone order\b", re.I)

# One class line of the report's accuracy-classes section:
#   "  fp64 all-orders       144 measured | fastest ... | not ordered"
CLASS_LINE = re.compile(r"^\s{2}([a-z0-9]+(?:-[a-z0-9]+)*)\s+([a-z-]+)\s+(\d+) measured", re.M)

# And the line of a class that produced no figure at all:
#   "  fp64 fixed-n          no option of this precision and shape produced a figure on this run"
EMPTY_LINE = re.compile(r"^\s{2}([a-z0-9]+(?:-[a-z0-9]+)*)\s+([a-z-]+)\s+no option of this precision",
                        re.M)

# A declared entry of one shape at one precision, read from the headers: the surface's own
# statement that the combination is one the library serves.
DECLARATION = re.compile(r"^\s*(?:inline\s+|extern\s+)?void\s+(Boys[A-Za-z0-9_]+)\s*\(", re.M)

# The axis line of the report's option-space section, which wraps over more than one line:
#   "axes: 2 route(s) (chebyshev, rational-minimax) | 2 scheme(s) (split-clenshaw, horner) | ..."
#   "      2 packing axis/axes (arguments, orders) | 3 division form(s) (...) | 2 exponential(s)"
# The member's own name is not read, so the pattern takes any run of lower-case words and
# separators up to the parenthesis that opens the member list.
AXIS_GROUP = re.compile(r"(\d+)\s+[a-z][a-z /-]*\s*\(")

# The entry each shape is asked through, per precision. The prefix is the library's own spelling;
# a class with no entry is a class the surface does not declare and this check does not count.
ENTRY = {
    "single": "BoysSingle",
    "all-orders": "BoysAllOrders",
    "fixed-n": "BoysFixedN",
    "all-n": "BoysAllN",
    "all-n-at-orders": "BoysAllNAtOrders",
}
SUFFIX = {"fp64": "", "fp32": "F32", "fp16": "F16", "bf16": "Bf16"}
SCALAR = {"fp64": "double", "fp32": "float", "fp16": "boys::F16", "bf16": "boys::Bf16"}
BUDGET = {"fp64": "kFloat", "fp32": "kFloat", "fp16": "kFp16", "bf16": "kFp16"}
VALUE = {"fp64": "1.5", "fp32": "1.5f", "fp16": "boys::F16(1.5f)", "bf16": "boys::Bf16(1.5f)"}
CALL = {
    "single": "boys::{entry}{suffix}<{policy}>(4, x);",
    "all-orders": "boys::{entry}{suffix}<{policy}>(4, x, out);",
    "fixed-n": "boys::{entry}{suffix}<{policy}>(4, xs, out, 2, 1);",
    "all-n": "boys::{entry}{suffix}<{policy}>(4, xs, out, 2);",
    "all-n-at-orders": "boys::{entry}{suffix}<{policy}>(ns, xs, out, 2);",
}


def one_order_shapes(reader: Reader) -> set[str]:
    """The Shape enumerators whose own documentation names one order."""
    text = reader.text("include/boys/boys.hpp")
    start = text.find("enum class Shape")

    if start < 0:
        return set()

    end = text.find("};", start)
    names = set()

    for match in SHAPE_ENUM.finditer(text[start:end]):
        if ONE_ORDER.search(match.group(2)):
            names.add(match.group(1))

    return names


def axes_of(report: str) -> list[int] | None:
    """The axis cardinalities the report states, in its own order.

    The block wraps, so the line that opens it is joined to the continuation lines the report
    indents under it, and the join stops at the line that closes it.
    """
    lines = report.splitlines()
    start = next((i for i, l in enumerate(lines) if l.strip().startswith("axes:")), None)

    if start is None:
        return None

    block = [lines[start]]

    for line in lines[start + 1:start + 6]:
        block.append(line)

        if "counted above" in line or not line.strip():
            break

    return [int(m.group(1)) for m in AXIS_GROUP.finditer(" ".join(block))]


def measured_by_class(report: str) -> dict[tuple[str, str], int]:
    """Every class the report states a measured count for, keyed by (precision, shape)."""
    counts: dict[tuple[str, str], int] = {}

    for match in CLASS_LINE.finditer(report):
        counts[(match.group(1), match.group(2))] = int(match.group(3))

    return counts


# The library target's own statement of what a consumer's translation unit needs defined:
#   target_compile_definitions(boys PUBLIC BoysFp16=1)
# ``boys.hpp`` gates the half-precision surface on ``BoysFp16`` and states no default for it, so a
# unit compiled without the definition is compiled against a smaller library than the one the
# report measured. Which definitions those are is read from the build file, not written here.
DEFINITIONS = re.compile(r"target_compile_definitions\(\s*boys\s([^)]*)\)", re.S)
DEFINITION_KEYWORD = re.compile(r"\b(PRIVATE|PUBLIC|INTERFACE)\b")
CONFIGURED = re.compile(r"\b(?:option|set|unset)\(\s*([A-Za-z_]\w*)")


def build_definitions(tree: pathlib.Path) -> tuple[list[str], list[str]]:
    """The compile definitions a consumer of the headers inherits, and the names left out.

    ``boys.hpp`` gates the half-precision surface on ``BoysFp16``, which the build states as a
    PUBLIC compile definition and the header states no default for: a unit compiled without it is a
    unit compiled against a smaller library than the one the report measured. Which definitions
    those are is read from the build file at the revision and never written here, and a name whose
    value the build file works out at configure time is named rather than guessed.
    """
    path = pathlib.Path(tree) / "CMakeLists.txt"
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    values: dict[str, set[str]] = {}
    unreadable: dict[str, str] = {}

    for call in DEFINITIONS.finditer(text):
        parts = DEFINITION_KEYWORD.split(call.group(1))

        for index in range(1, len(parts) - 1, 2):
            if parts[index] not in ("PUBLIC", "INTERFACE"):
                continue

            for token in re.split(r"[\s;]+", parts[index + 1]):
                name = token.split("=", 1)[0]

                if not re.fullmatch(r"[A-Za-z_]\w*", name) or "$" in token or '"' in token:
                    unreadable.setdefault(name, "its value is worked out at configure time")
                    continue

                values.setdefault(name, set()).add(token)

    # A name the build file sets, options or clears governs the definitions spelled under it: a
    # definition the file states inside the branch a cache entry decides is a definition the
    # shipped configure does not carry, and this check read no configure.
    for name in sorted(values):
        governing = [c for c in sorted({m.group(1) for m in CONFIGURED.finditer(text)})
                     if name == c or name.startswith(c + "_")]

        if governing:
            unreadable.setdefault(
                name, f"the build file decides it at configure time, under {governing[0]}")

    carried: list[str] = []
    skipped: list[str] = []

    for name in sorted(values):
        if name in unreadable:
            skipped.append(f"{name} ({unreadable[name]})")
        elif len(values[name]) > 1:
            skipped.append(f"{name} (the build file states "
                           f"{len(values[name])} different values, one per configure branch)")
        else:
            carried.append(sorted(values[name])[0])

    return carried, skipped


def compiled(tree: pathlib.Path, source: str, definitions: list[str]) -> bool:
    """Whether one scratch translation unit compiles. The negative control of this check.

    The unit is compiled inside a tree exported from the revision under test, so the compiler never
    sees another writer's half-finished edit to the library this check is asserting about, and it
    carries the definitions the library's own target publishes to its consumers, so the compiler
    sees the surface the report measured rather than a smaller one.
    """
    with tempfile.TemporaryDirectory() as scratch:
        unit = pathlib.Path(scratch) / "probe.cpp"
        unit.write_text(source, encoding="utf-8")
        log = pathlib.Path(scratch) / "log.txt"
        flags = " ".join(f"/D{definition}" for definition in definitions)
        script = pathlib.Path(scratch) / "run.bat"
        script.write_text(
            "@echo off\n"
            'call "C:\\Program Files\\Microsoft Visual Studio\\18\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat" >nul 2>&1\n'
            f'cd /d "{tree}"\n'
            f'cl /nologo /std:c++latest /EHsc /Zs {flags} /I include "{unit}" > "{log}" 2>&1\n',
            encoding="utf-8",
        )

        try:
            completed = subprocess.run(["cmd.exe", "/c", str(script)], capture_output=True,
                                       timeout=900)
        except subprocess.TimeoutExpired:
            # Not a refusal: the compiler never answered. The control reads a False in the
            # arguments-axis position as the exclusion failing, so a deadline that expired is
            # printed as itself rather than left to look like the library's answer.
            print("      the compiler did not answer before this unit's deadline")
            return False
        except FileNotFoundError:
            return False

        if completed.returncode == 0:
            return True

        log_text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""

        for line in [l for l in log_text.splitlines()
                     if " error " in l or "fatal error" in l][:2]:
            print(f"      {line.strip()[:160]}")

        return False


def policy(precision: str, pack: str) -> str:
    return (f"boys::EvalPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kHorner, "
            f"boys::BoysBudget::{BUDGET[precision]}, boys::PackAxis::{pack}, "
            f"boys::FitGranularity::kNarrow, boys::DivisionForm::kRefinedReciprocal, "
            f"boys::RegionBExp::kFast>")


def unit_for(precision: str, shape: str, pack: str) -> str:
    call = CALL[shape].format(entry=ENTRY[shape], suffix=SUFFIX[precision],
                              policy=policy(precision, pack))
    return ("#include <boys/boys.hpp>\n"
            "void Instantiate() {\n"
            f"    [[maybe_unused]] {SCALAR[precision]} x = {VALUE[precision]};\n"
            f"    [[maybe_unused]] {SCALAR[precision]} xs[8] = {{}};\n"
            f"    [[maybe_unused]] {SCALAR[precision]} out[64] = {{}};\n"
            "    [[maybe_unused]] int ns[8] = {};\n"
            f"    {call}\n"
            "}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="the option probe's report")
    parser.add_argument("--root", default=".")
    parser.add_argument("--rev", default="HEAD",
                        help="the revision the library is read at (never the working tree)")
    parser.add_argument("--no-compile", action="store_true",
                        help="skip the compilation control (the check then asserts, not proves)")
    arguments = parser.parse_args()
    root = pathlib.Path(arguments.root).resolve()
    reader = Reader(root, arguments.rev)

    print(f"library read at {arguments.rev} ({reader.rev_id()}), never from the working tree")

    report = pathlib.Path(arguments.report).read_text(encoding="utf-8", errors="replace")
    cardinalities = axes_of(report)

    if not cardinalities or len(cardinalities) != 6:
        print(f"check_class_combinations: the report states {len(cardinalities or [])} axis "
              "cardinalit(ies) where the policy has six - the report is not one this check reads")
        return 1

    full = 1
    for size in cardinalities:
        full *= size

    one_order = one_order_shapes(reader)
    measured = measured_by_class(report)

    if not measured:
        print("check_class_combinations: the report states no class with a measured count, so this "
              "run checked nothing and its exit status is a failure rather than a clean bill")
        return 1

    if not one_order:
        print("check_class_combinations: no Shape enumerator's documentation names one order, so "
              "the exclusion this check rests on was read as absent - the reader is wrong, not the "
              "library")
        return 1

    print(f"axes: {' x '.join(str(c) for c in cardinalities)} = {full} combination(s) per class")
    print(f"one-order shapes, read from the library: "
          f"{', '.join(sorted(one_order)) if one_order else 'none'}")

    # The control: a one-order shape cannot take the orders axis. One instantiation each way, in a
    # tree exported from the revision, so what the compiler reads is the revision and not a working
    # tree another writer may hold.
    if not arguments.no_compile:
        tree = reader.export()
        definitions, not_carried = build_definitions(tree)

        if not definitions:
            print("check_class_combinations: the revision's build file states no PUBLIC compile "
                  "definition for the library target that this check can read, so a unit compiled "
                  "here would be compiled against a surface it cannot claim is the report's - the "
                  "reader is wrong, not the library")
            return 1

        print(f"compiling the revision exported to {tree}, carrying the build's own "
              f"definition(s): {', '.join(definitions)}")

        if not_carried:
            print("  not carried, and named rather than guessed: " + "; ".join(not_carried))

        for shape_id in sorted(one_order):
            token = shape_id[1:].lower()
            shapes = [s for s in ENTRY if s.replace("-", "") == token]

            if not shapes:
                continue

            shape = shapes[0]
            at_orders = compiled(tree, unit_for("fp32", shape, ORDERS_AXIS), definitions)
            off_orders = compiled(tree, unit_for("fp32", shape, "kArguments"), definitions)

            if at_orders or not off_orders:
                print(f"  [{shape}] control: at the orders axis compiles = {at_orders}, "
                      f"without it compiles = {off_orders} - the exclusion this check rests on "
                      "does not hold, so its numbers are not admissible")
                return 1

            print(f"  [{shape}] control: the orders axis is refused by the compiler and the "
                  "arguments axis is accepted")

    # The surface's own statement of what exists, so a class the report leaves empty is held to
    # it: a class with no figure whose entry the headers declare is the same finding as a class
    # measured short of its combinations, and one whose entry they do not declare is a gap in
    # the library that this check names rather than counts against the probe.
    surface = set()
    for header in ("boys.hpp", "boys_span.hpp", "boys_half.hpp"):
        surface.update(DECLARATION.findall(reader.text(f"include/boys/{header}")))

    if not surface:
        print("check_class_combinations: the headers declare no entry at the revision under test, "
              "so the surface was read as empty - the reader is wrong, not the library")
        return 1

    def is_one_order(shape: str) -> bool:
        return shape.replace("-", "").lower() in {name[1:].lower() for name in one_order}

    def entry_name(precision: str, shape: str) -> str:
        return ENTRY[shape] + SUFFIX[precision]

    empty = {(p, s) for p, s in EMPTY_LINE.findall(report)}

    shortfall = 0
    undeclared = []
    print("\n  class                     possible  measured  shortfall")

    for precision, shape in sorted(set(measured) | empty):
        if shape not in ENTRY:
            continue

        count = measured.get((precision, shape), 0)

        if count == 0 and entry_name(precision, shape) not in surface:
            # No entry, so the class holds no combination anything could be instantiated at. It is
            # named rather than counted: a class the library does not serve is a hole in the
            # surface, and a zero in the possible column is what that is.
            undeclared.append((precision, shape))
            print(f"  {precision + ' ' + shape:<24} {0:>8}  {0:>8}  {'no entry':>9}")
            continue

        # A one-order shape drops the orders member of the packing axis: the axis has one member
        # there and not two, so the product loses exactly the factor the axis contributed.
        pack_index = 3  # route, scheme, partition, packing, division, exponential
        expected = full // cardinalities[pack_index] if is_one_order(shape) else full

        gap = expected - count
        shortfall += gap if gap > 0 else 0
        print(f"  {precision + ' ' + shape:<24} {expected:>8}  {count:>8}  {gap:>9}")

    if undeclared:
        print(f"\n{len(undeclared)} class(es) have no entry in the headers: "
              + ", ".join(f"{p} {s}" for p, s in undeclared)
              + " — the library declares no such shape at that precision, so no combination "
                "of it can be measured and the gap is the surface's rather than this run's.")

    print()
    print(HOST_TOTAL.format(total=shortfall))

    if shortfall:
        print(f"\n{shortfall} combination(s) a class can be instantiated at and this run did not "
              "measure. A class's count is its own: every member of every axis it offers is a "
              "combination the entry is instantiated at, and a report that states fewer is an "
              "account of the probe rather than of the library.")
        return 1

    print("\nevery class's measured count is the number of combinations it can be instantiated at")
    return 0


if __name__ == "__main__":
    sys.exit(main())
