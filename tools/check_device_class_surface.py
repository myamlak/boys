#!/usr/bin/env python3
r"""Check that every device class the device surface's enumerations imply is carried.

The device surface is a cross of two enumerations the library declares in
`include/boys/boys_cuda_options.hpp`: `DeviceOptionPrecision` and `DeviceOptionQuestion`. A class
is one cell of that cross, and a class with no row is a combination a caller can name and the
library cannot serve.

`tools/check_device_entry_bijection.py` reconciles the `DeviceEntry` rows against the members
`boys_cuda.hpp` declares, and its unit is the entry. That unit cannot see this defect: a row list
that carries one precision's rows and none of another precision's reads complete from inside
itself, and the entry every one of its rows names is present. The precision is served and the
class is not, and the two look identical to that check. That is the hole this check exists to
close, on the device as `tools/check_class_surface.py` closes it on the host.

**The class axis is the question and not the row's own shape.** The library's own count of the
classes a run must print is taken over `(row.precision, row.question)`
(`src/boys_cuda_probe.cpp`, the closure's slot), and the probe's build-defaults emission keys a
device row's shape cell on a `DeviceOptionQuestion` (`SeamShapeCell`) - and states the number it
keys: "the nine classes of the device half" (`boys_cuda_probe.hpp`). The two `DeviceOptionShape`
members `kAllOrders` and `kEachOrder` are one question and not two, which that header says of the
question ("the shape with the sink/array difference folded out: the two are one question and not
two"), so a cross of the shape would count twelve classes where this surface has nine.

**What carries a class.** A class is carried by a row of the option table (`src/boys_cuda.cpp`,
`kDeviceOptions`) whose own precision and question are that class; it is refused with a stated
reason when every row of it refuses the build and states why, which is `DeviceOptionInfo`'s
`refusedBecause` ("why not, when built is false; nullptr otherwise"). The reason is read from the
row and not from a list beside it: the rule this surface states for its own fields is that a fact
is read from the row's own entry and never listed beside it (`DeviceOptionInfo`,
boys_cuda_options.hpp), and a second hand-written statement of one fact is the shape that
disagrees with the code without anything reading it - which is the defect this check exists to
close.

**The reading is textual, as the host check's is.** A row whose `built` cell names a build-time
constant rather than a literal counts as carried, and whether a configuration serves it is the
row's own statement: `src/boys_cuda.cpp` carries both values of the fp16 seam with the refusal
text on the closed branch, so the class is carried or refused-with-a-reason in either
configuration and never silent. Nothing here is compiled and no configuration is evaluated.

**The direction that matters.** A class this check finds no row for and no reason for is a defect,
and a reason is admissible only when it is a proved impossibility rather than work not done. The
other direction is read too: a row whose precision and question are no class of the cross is a row
for a class the enumerations do not name.

Comment lines and trailing comments are stripped before the search, because a row that was
commented out is not a row the library carries.

Usage:

    python3 tools/check_device_class_surface.py --check
    python3 tools/check_device_class_surface.py --options <file> --rows <file>   # negative control
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import re
import sys
from typing import NamedTuple

REPO = pathlib.Path(__file__).resolve().parent.parent
OPTIONS = REPO / "include" / "boys" / "boys_cuda_options.hpp"
ROWS = REPO / "src" / "boys_cuda.cpp"

# The two enumerations the cross is read from, both in the options header.
PRECISION_AXIS = "DeviceOptionPrecision"
QUESTION_AXIS = "DeviceOptionQuestion"

# The sentinel both axes of this surface end with, on that header's own words: "one past the last".
# It is excluded from the walk and its absence stops the run rather than being guessed at - the
# exclusion is by name, so an axis whose sentinel this reading cannot find would be walked as
# classes. The spelling is the one tools/check_enum_arms_covered.py gives the same sentinel.
SENTINEL = "kCount"

# One enumeration's body, and one enumerator of it: an optional initialiser is skipped so that the
# first enumerator of an axis - the only one written `= 0` - reads like the rest.
ENUM = r"enum class {name}\s*:\s*[A-Za-z_][\w:]*\s*\{{(.*?)\n\}};"
ENUMERATOR = re.compile(r"^\s*(k[A-Za-z0-9]+)\s*(?:=[^,]*)?,", re.MULTILINE)

# The option table: the array the library's own report is a projection of, and one row of it. A row
# carries no nested braces, so its body is everything up to its closing brace.
ROW_TABLE = re.compile(r"constexpr DeviceOptionInfo kDeviceOptions\[\]\s*=\s*\{(.*?)\n\};",
                       re.DOTALL)
ROW = re.compile(r"\{DeviceEntry::(\w+)\s*,(.*?)\}\s*,", re.DOTALL)

# A cell of a row: an enumerator or a constant, or a string literal. The last two cells of a row
# are `built` and `refusedBecause`; every field after them is defaulted from the entry.
CELL = r'(?:[A-Za-z_]\w*|"(?:[^"\\]|\\.)*")'
TAIL = re.compile(rf"({CELL})\s*,\s*({CELL})\s*$")

PRECISION = re.compile(r"DeviceOptionPrecision::(\w+)")
QUESTION = re.compile(r"DeviceOptionQuestion::(\w+)")


class Row(NamedTuple):
    """One row of the option table, as the four cells this check reads it by."""

    entry: str
    precision: str
    question: str
    built: str
    refused: str


class Class(NamedTuple):
    """One cell of the cross: the precision lane and the question an entry of it answers."""

    precision: str
    question: str


def code_only(text: str) -> str:
    """`text` with every comment line and trailing comment removed.

    A row that was commented out is not a row the library carries, so prose must not be allowed to
    answer for one. Block comments are not stripped because this translation unit carries none.
    """
    kept = []

    for line in text.splitlines():
        stripped = line.lstrip()

        if stripped.startswith("//"):
            continue

        kept.append(line.split("//", 1)[0])

    return "\n".join(kept)


def axis_members(text: str, name: str) -> list[str]:
    """The classes' members of `name`: every enumerator it declares but the sentinel it ends with."""
    match = re.search(ENUM.format(name=name), text, re.DOTALL)

    if match is None:
        raise SystemExit(f"check_device_class_surface: the options header declares no enum class "
                         f"{name}, so no class is read from it. A check that can pass by reading "
                         f"nothing is not a check")

    members = ENUMERATOR.findall(match.group(1))

    if not members:
        raise SystemExit(f"check_device_class_surface: {name} declares no enumerator: a class "
                         f"walked over nothing is not a check")

    if members[-1] != SENTINEL:
        raise SystemExit(
            f"check_device_class_surface: {name} does not end in {SENTINEL} - its last enumerator "
            f"is {members[-1]}. The sentinel is excluded by name, so it is named here rather than "
            f"worked around: walked as it stands, this axis would report the sentinel as a class "
            f"with no row, or a class as the sentinel")

    return members[:-1]


def rows_of(path: pathlib.Path) -> list[Row]:
    """Every row of the option table, as the cells this check reads it by."""
    text = code_only(path.read_text(encoding="utf-8", errors="replace"))
    table = ROW_TABLE.search(text)

    if table is None:
        raise SystemExit(f"check_device_class_surface: {path} declares no kDeviceOptions array: "
                         f"the option table this check walks is not in it")

    found = []

    for entry, body in ROW.findall(table.group(1)):
        precision = PRECISION.search(body)
        question = QUESTION.search(body)
        tail = TAIL.search(body.rstrip().rstrip(","))

        if precision is None or question is None or tail is None:
            raise SystemExit(
                f"check_device_class_surface: the row {entry} in {path} does not read as a row of "
                f"this table - its precision, its question or the built/refused pair is missing. "
                f"A row read short is a class this check would report as uncarried")

        found.append(Row(entry, precision.group(1), question.group(1), tail.group(1),
                         tail.group(2)))

    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero on a class the cross names and no row carries and no "
                             "row refuses with a stated reason; this is the form CI runs")
    parser.add_argument("--options", type=pathlib.Path, default=OPTIONS,
                        help="the header the two axes are read from")
    parser.add_argument("--rows", type=pathlib.Path, default=ROWS,
                        help="the translation unit the option table is read from")
    args = parser.parse_args()

    if not args.options.is_file() or not args.rows.is_file():
        print(f"check_device_class_surface: missing {args.options} or {args.rows}", file=sys.stderr)
        return 1

    text = args.options.read_text(encoding="utf-8", errors="replace")
    precisions = axis_members(text, PRECISION_AXIS)
    questions = axis_members(text, QUESTION_AXIS)
    rows = rows_of(args.rows)

    if not rows:
        print(f"check_device_class_surface: read {len(precisions)} precision(s), "
              f"{len(questions)} question(s) and no row of the option table from {args.rows}. A "
              f"check that can pass by reading nothing is not a check", file=sys.stderr)
        return 1

    cross = [Class(p, q) for p in precisions for q in questions]
    named = set(cross)
    by_class: dict[Class, list[Row]] = collections.defaultdict(list)
    outside: list[Row] = []

    for row in rows:
        key = Class(row.precision, row.question)

        if key not in named:
            outside.append(row)
            continue

        by_class[key].append(row)

    carried = {key: [row for row in held if row.built != "false"] for key, held in by_class.items()}
    stated = {key: [row for row in held if row.built == "false" and row.refused != "nullptr"]
              for key, held in by_class.items()}

    print(f"check_device_class_surface: {len(precisions)} device precision(s) x "
          f"{len(questions)} question(s) = {len(cross)} class(es), over {len(rows)} row(s) of the "
          f"option table")
    print(f"  carried by a row      {sum(1 for key in cross if carried.get(key))}")
    print(f"  refused, with reason  {sum(1 for key in cross if not carried.get(key) and stated.get(key))}")

    for key in cross:
        if not carried.get(key) and stated.get(key):
            first = stated[key][0]
            print(f"    {key.precision} x {key.question}: every one of its "
                  f"{len(by_class[key])} row(s) refuses this build; {first.entry} states "
                  f"{first.refused}")

    missing = [key for key in cross if not carried.get(key) and not stated.get(key)]
    failures = len(missing)

    if missing:
        print(f"  with NO row and NO stated reason: {len(missing)}", file=sys.stderr)

    for key in missing:
        held = by_class.get(key, [])

        if not held:
            print(f"check_device_class_surface: ({key.precision}, {key.question}) has no row: no "
                  f"row of the option table in {args.rows} carries this precision and question, "
                  f"and no row that refuses the class states a reason for it. A class the "
                  f"library's own enumerations imply is either served or refused with a proved "
                  f"impossibility, and work not done is not one.", file=sys.stderr)
        else:
            names = ", ".join(row.entry for row in held)
            print(f"check_device_class_surface: ({key.precision}, {key.question}) has "
                  f"{len(held)} row(s) and every one refuses this build with no stated reason: "
                  f"{names}. An option the library does not carry is refused where it is named, "
                  f"with the reason, and a refusal nobody stated is work not done.",
                  file=sys.stderr)

    if outside:
        failures += len(outside)
        print(f"\ncheck_device_class_surface: {len(outside)} row(s) are of no class the "
              f"enumerations name:", file=sys.stderr)

        for row in outside:
            print(f"  {row.entry} carries {row.precision} and {row.question}, and the cross is "
                  f"{' x '.join(name for name in (PRECISION_AXIS, QUESTION_AXIS))} - a row of no "
                  f"class is an option nothing can ask for", file=sys.stderr)

    if failures:
        return 1

    print(f"\ncheck_device_class_surface: all {len(cross)} class(es) the "
          f"{PRECISION_AXIS}-by-{QUESTION_AXIS} cross names are carried by a row or refused with a "
          f"stated reason")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
