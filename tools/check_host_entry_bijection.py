#!/usr/bin/env python3
r"""Check that every host entry the headers declare has one shape, and every shape an entry.

The host surface is described by the declarations in three headers and by `tools/host-shapes.json`,
and nothing reconciled them. The device surface's twin check found a real defect, in the direction
that matters most: rows naming an entry that had never been declared, which four mechanisms then
consumed without one of them asking whether the entry was there. The host table is hand-written and
has the same exposure - a shape nothing reaches reads exactly like a shape this surface has.

This check is the bijection between the two lists a caller meets: the names a header declares, and
the shapes the table places them in. Both directions are defects:

  * a declared name the table does not place is a call whose shape nothing states, so the next
    reader of the surface cannot tell what it is a call to;
  * a name the table places that no header declares is a shape described for a call that does not
    exist, which a later reader takes for a decision.

The unit is the shape, not the name. A run-time tier, a route-named sibling, a sorted-argument
overload and a span overload are spellings of one shape, so the table maps names to shape ids and
lists the aliases nowhere; and what a shape offers is DERIVED here from its tuple rather than
written in the file, so a one-order shape has a packing axis of one value by construction rather
than by a refusal in the table.

Names are read from the headers rather than from a second list, so the check cannot agree with
itself while disagreeing with the code. Two things are skipped on the way, both because they spell
an entry's name without being an entry: the `extern template` instantiations in boys.hpp's `\cond`
block - instantiations at the defaults, not declarations - and every line inside that block.

Usage:
  python3 tools/check_host_entry_bijection.py --check
  python3 tools/check_host_entry_bijection.py --table <file> [--header <file>]   # negative control
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
TABLE = pathlib.Path(__file__).resolve().parent / "host-shapes.json"
SOURCES = (
    REPO / "include" / "boys" / "boys.hpp",
    REPO / "include" / "boys" / "boys_span.hpp",
    REPO / "include" / "boys" / "boys_c.h",
)

# The tuple's vocabulary, which the table also states: the file documents what each axis value
# means, and reading it back here is what keeps the documentation from drifting off the code.
VOCABULARY = {
    "orders": ("one", "ladder"),
    "args": ("scalar", "array"),
    "extent": ("single", "call", "per_element"),
    "cardinality": ("one", "nmax+1", "count", "count*(nmax+1)", "count*(1+max n[i])"),
    "protocol": ("returned", "out", "status_out"),
    "host": ("host", "device"),
}

SHAPE_KEYS = {"id", "tuple", "evidence"}
ENTRY_KEYS = {"shape", "spelling"}
TOP_KEYS = {"format", "why", "sources", "vocabulary", "shapes", "entries", "not_an_evaluation"}
FORMAT = "boys.host-shapes/1"

# A declaration: the name the line declares is the last identifier before the call's open paren, so
# anchoring on the paren keeps a return type, a template parameter and a member's own name out of
# the match. Only column 0 is read, so a continued parameter list is not a second declaration.
DECLARATION = re.compile(r"^[A-Za-z_][A-Za-z0-9_:<>,\s\*&]*?\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")


class DuplicateKey(ValueError):
    """A key written twice in the JSON object: one name placed in two shapes."""


def declared(path: pathlib.Path) -> list[tuple[str, int]]:
    """Every name `path` declares at namespace scope, as (name, line)."""
    found: list[tuple[str, int]] = []
    hidden = False
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()

    for number, line in enumerate(text, 1):
        if "\\cond" in line:
            hidden = True
        if "\\endcond" in line:
            hidden = False
            continue
        if hidden or line.startswith("extern template"):
            continue

        match = DECLARATION.match(line)

        if match is not None:
            found.append((match.group(1), number))

    return found


def cardinality_of(orders: str, args: str, extent: str) -> str:
    """The output cardinality a tuple derives, which is what the table's own field must say."""
    if orders == "one":
        return "one" if args == "scalar" else "count"
    if extent == "per_element":
        return "count*(1+max n[i])"
    return "nmax+1" if args == "scalar" else "count*(nmax+1)"


def derived(tuple_: dict) -> dict:
    """What a shape's axes are, from its tuple alone.

    This is the pure function the table is kept out of: a caller reading the table sees the axes a
    shape has without the table being able to state one the tuple does not carry.
    """
    orders, args, extent = tuple_["orders"], tuple_["args"], tuple_["extent"]

    # PackAxis::kOrders holds four orders of one argument, so it needs a ladder; kArguments holds
    # four arguments of one order, so it needs an array. A shape with neither has no wide dimension
    # for a lane to fill and its axis space is the enumeration's single default.
    axes = []
    if orders == "ladder":
        axes.append("kOrders")
    if args == "array":
        axes.append("kArguments")

    if orders == "one":
        per_argument = 1
    else:
        per_argument = "nmax+1" if extent == "call" else "1+max n[i]"

    return {
        "values_per_argument": per_argument,
        "output_elements": tuple_["cardinality"],
        "packing_axes": axes,
        "packing_axis_cardinality": len(axes) if axes else 1,
    }


def load_table(path: pathlib.Path) -> dict:
    """The table, with a key written twice refused rather than silently kept last."""

    def pairs_hook(pairs: list[tuple[str, object]]) -> dict:
        seen: dict = {}
        for key, value in pairs:
            if key in seen:
                raise DuplicateKey(
                    f"{key} is written twice, so one name is placed in two shapes and only the "
                    f"last placement is read")
            seen[key] = value
        return seen

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs_hook)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="the CI spelling; a disagreement exits non-zero either way")
    parser.add_argument("--table", type=pathlib.Path, default=TABLE, help="the shape table")
    parser.add_argument("--header", type=pathlib.Path, action="append", default=None,
                        help="a header the names are read from; repeats, replaces the default set")
    args = parser.parse_args()

    sources = tuple(args.header) if args.header else SOURCES
    failures = 0

    for path in (args.table, *sources):
        if not path.is_file():
            print(f"check_host_entry_bijection: missing {path}", file=sys.stderr)
            return 1

    try:
        table = load_table(args.table)
    except (DuplicateKey, json.JSONDecodeError, OSError) as error:
        print(f"check_host_entry_bijection: {args.table} does not read: {error}", file=sys.stderr)
        return 1

    # The declarations, per source and together. A source that yields nothing is a parse that went
    # wrong rather than a surface with no entries, and it is reported before anything else: a check
    # that can pass by reading nothing is not a check.
    per_source: list[tuple[pathlib.Path, list[tuple[str, int]]]] = [
        (path, declared(path)) for path in sources]

    for path, names in per_source:
        if not names:
            print(
                f"check_host_entry_bijection: {path} declares nothing this reader can see. Either "
                f"the file moved or its declarations are no longer where they were read from",
                file=sys.stderr,
            )
            failures += 1

    if failures:
        return 1

    declared_names = {name for _, names in per_source for name, _ in names}

    # The table's own schema: unknown keys are refused rather than ignored, because a key this
    # check does not read is a claim about a shape that nothing holds to the code.
    for key in table:
        if key not in TOP_KEYS:
            print(f"check_host_entry_bijection: {args.table} carries the unknown key '{key}'",
                  file=sys.stderr)
            failures += 1

    if table.get("format") != FORMAT:
        print(f"check_host_entry_bijection: {args.table} is format {table.get('format')!r}, not "
              f"{FORMAT!r}", file=sys.stderr)
        failures += 1

    for axis, allowed in VOCABULARY.items():
        stated = table.get("vocabulary", {}).get(axis)
        if stated is not None and stated != list(allowed):
            print(f"check_host_entry_bijection: {args.table} documents the '{axis}' axis as "
                  f"{stated} where this check reads {list(allowed)}", file=sys.stderr)
            failures += 1

    shapes: dict[str, dict] = {}
    tuples: dict[tuple, str] = {}

    for row in table.get("shapes", []):
        unknown = sorted(set(row) - SHAPE_KEYS)
        shape = row.get("id", "")
        offends = failures

        if unknown:
            print(f"check_host_entry_bijection: shape '{shape}' carries {unknown}. An axis is "
                  f"derived from the tuple, so a stated one is a claim nothing holds to it",
                  file=sys.stderr)
            failures += 1
            continue

        if shape in shapes:
            print(f"check_host_entry_bijection: shape '{shape}' is written twice", file=sys.stderr)
            failures += 1
            continue

        tuple_ = row.get("tuple", {})

        for axis, allowed in VOCABULARY.items():
            if tuple_.get(axis) not in allowed:
                print(f"check_host_entry_bijection: shape '{shape}' has {axis}="
                      f"{tuple_.get(axis)!r}, which is not one of {list(allowed)}",
                      file=sys.stderr)
                failures += 1

        if not row.get("evidence"):
            print(f"check_host_entry_bijection: shape '{shape}' states no evidence. A tuple is a "
                  f"claim about a call, and the call it was read from belongs beside it",
                  file=sys.stderr)
            failures += 1

        if failures != offends:
            continue

        order_axis, arg_axis = tuple_["orders"], tuple_["args"]
        extent = tuple_["extent"]

        # Two of the tuple's own combinations are calls no signature can have, so they are refused
        # here rather than derived into a row nothing can be placed in.
        if (order_axis == "one") != (extent == "single"):
            print(f"check_host_entry_bijection: shape '{shape}' is orders={order_axis} with "
                  f"extent={extent}: one order has no ladder top, and a ladder has one",
                  file=sys.stderr)
            failures += 1
        if extent == "per_element" and arg_axis != "array":
            print(f"check_host_entry_bijection: shape '{shape}' takes per-element tops with a "
                  f"single argument: the tops arrive one per argument, as an array",
                  file=sys.stderr)
            failures += 1

        wanted = cardinality_of(order_axis, arg_axis, extent)
        if tuple_["cardinality"] != wanted:
            print(f"check_host_entry_bijection: shape '{shape}' states cardinality="
                  f"{tuple_['cardinality']} where its own orders/args/extent derive {wanted}",
                  file=sys.stderr)
            failures += 1

        key = tuple((axis, tuple_[axis]) for axis in VOCABULARY)
        if key in tuples:
            print(f"check_host_entry_bijection: shapes '{tuples[key]}' and '{shape}' carry one "
                  f"tuple: one call under two ids, so a name moved between them would still "
                  f"read as placed", file=sys.stderr)
            failures += 1

        tuples[key] = shape
        shapes[shape] = row

    if failures:
        return 1

    entries = table.get("entries", {})
    exempt = table.get("not_an_evaluation", {})
    reached: dict[str, list[str]] = {shape: [] for shape in shapes}

    for name, row in entries.items():
        if isinstance(row, str):
            row = {"shape": row}
        unknown = sorted(set(row) - ENTRY_KEYS)
        if unknown:
            print(f"check_host_entry_bijection: the row for {name} carries {unknown}",
                  file=sys.stderr)
            failures += 1
            continue
        if row.get("shape") not in shapes:
            print(f"check_host_entry_bijection: {name} names the shape {row.get('shape')!r}, which "
                  f"this table does not carry", file=sys.stderr)
            failures += 1
            continue
        reached[row["shape"]].append(name)

    both = sorted(set(entries) & set(exempt))
    for name in both:
        print(f"check_host_entry_bijection: {name} is placed in a shape and exempted from having "
              f"one: an entry is one or the other", file=sys.stderr)
        failures += 1

    unplaced = sorted(declared_names - set(entries) - set(exempt))
    for name in unplaced:
        where = next(path for path, names in per_source if any(n == name for n, _ in names))
        line = next(number for _, names in per_source for n, number in names if n == name)
        print(f"check_host_entry_bijection: {name}, declared in {where}:{line}, has no row "
              f"here: no shape claims it and it is not exempted, so what a call to it is is "
              f"unstated", file=sys.stderr)

    if unplaced:
        failures += len(unplaced)

    undeclared = sorted((set(entries) | set(exempt)) - declared_names)
    for name in undeclared:
        kind = "entries" if name in entries else "not_an_evaluation"
        print(f"check_host_entry_bijection: {kind}['{name}'] is in {args.table} and no header "
              f"declares {name}: the table describes a call that does not exist", file=sys.stderr)

    if undeclared:
        failures += len(undeclared)

    for shape in sorted(shapes):
        if not reached[shape]:
            print(f"check_host_entry_bijection: shape '{shape}' is reached by no entry: either the "
                  f"last name off it was moved and this row is what the surface no longer has, or "
                  f"it was written before the entry that belongs in it", file=sys.stderr)
            failures += 1

    if failures:
        return 1

    print("check_host_entry_bijection: the host surface")
    print()
    for path, names in per_source:
        distinct = len({name for name, _ in names})
        shown = path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else str(path)
        print(f"  {shown}: {len(names)} declaration(s), {distinct} distinct name(s)")
    print(f"  declared: {len(declared_names)} distinct name(s) held against {len(shapes)} shape(s)")
    print(f"  table:    {len(entries)} name(s) placed in a shape, {len(exempt)} exempted with a "
          f"reason")

    print()
    for shape in sorted(shapes, key=lambda s: (-len(reached[s]), s)):
        tuple_ = shapes[shape]["tuple"]
        axes = derived(tuple_)
        stated = ", ".join(f"{axis}={tuple_[axis]}" for axis in VOCABULARY)
        packed = axes["packing_axes"] or ["none: the enumeration's default is all of it"]
        print(f"  {shape}  ({len(reached[shape])} name(s))")
        print(f"    {stated}")
        print(f"    derived: {axes['values_per_argument']} value(s) per argument, "
              f"{axes['output_elements']} output element(s), packing axis {packed}")

    print()
    print(f"check_host_entry_bijection: the {len(declared_names)} declared name(s) are placed, "
          f"each in one shape, and every shape is reached")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
