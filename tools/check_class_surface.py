#!/usr/bin/env python3
r"""Check that every class the library's own enumerations imply has an entry.

The host surface is a cross of two enumerations the library declares: `Precision` and `Shape`, both
in `include/boys/boys.hpp`. A class is one cell of that cross, and a class with no entry is a
combination a caller can name and the library cannot serve.

`tools/check_host_entry_bijection.py` reconciles the names a header declares with the shapes the
table places them in, and its unit is the shape. That unit cannot see this defect: a shape counts as
placed when ANY precision has an entry for it, so `Shape::kFixedN` is satisfied by the double lane's
`BoysFixedN` alone while the single-precision and half lanes carry no entry at that shape at all.
The shape is served and the class is not, and the two look identical to that check. That is the hole
this check exists to close: a class the library's enumerations imply and its headers do not declare
is reported here, whichever precision is missing it.

**The direction that matters.** A class this check does not find an entry for is a defect unless it
is stated in the table's `absent` list with a reason, and a reason is admissible only when it is a
proved impossibility rather than work not done. An empty `absent` list therefore means "the cross is
served", which is the condition this project merges on.

The check reads one thing: an entry whose default policy names the class, in non-comment code. A
class named only in a comment is not served, so comment lines and trailing comments are stripped
before the search rather than being allowed to answer for an entry that is not there.

Usage:

    python3 tools/check_class_surface.py --check
"""

import argparse
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
HEADER = REPO / "include" / "boys" / "boys.hpp"
TABLE = pathlib.Path(__file__).resolve().parent / "host-shapes.json"

# A class is named by an entry's own default policy: `DefaultPolicy<Precision::kX, Shape::kY>`, the
# same key the row list is written from and the same one the entries' template defaults spell.
CLASS = re.compile(r"DefaultPolicy<\s*Precision::(k[A-Za-z0-9]+)\s*,\s*Shape::(k[A-Za-z0-9]+)\s*>")

# The two enumerations, read from the header rather than written here: an enumerator added to either
# is a class this check grows, which is the point of reading them.
ENUM = r"enum class {name}\s*:\s*std::uint8_t\s*\{{(.*?)\n\}};"
ENUMERATOR = re.compile(r"^\s*(k[A-Za-z0-9]+)\s*(?:=[^,]*)?,", re.MULTILINE)


def enumerators(text: str, name: str) -> list[str]:
    """The enumerator names `name` declares, in declaration order."""
    match = re.search(ENUM.format(name=name), text, re.DOTALL)

    if match is None:
        raise SystemExit(f"check_class_surface: the header declares no enum class {name}")

    return ENUMERATOR.findall(match.group(1))


def code_only(text: str) -> str:
    """`text` with every comment line and trailing comment removed.

    A class named only in prose is not served, so prose must not be allowed to answer for an entry
    that is not there. Block comments are not stripped because this header does not use them for
    anything a class could be named in; a `\\cond` region is stripped by the caller.
    """
    kept = []

    for line in text.splitlines():
        stripped = line.lstrip()

        if stripped.startswith("///") or stripped.startswith("//"):
            continue

        kept.append(line.split("//", 1)[0])

    return "\n".join(kept)


def visible(text: str) -> str:
    """`text` with the `\\cond` spans removed, which is the same region the bijection check skips."""
    kept = []
    hidden = False

    for line in text.splitlines():
        if "\\cond" in line:
            hidden = True
        if "\\endcond" in line:
            hidden = False
            continue
        if not hidden:
            kept.append(line)

    return "\n".join(kept)


def absent_reasons(table_path: pathlib.Path) -> dict:
    """The classes the table states a reason for, as {(precision, shape): reason}."""
    table = json.loads(table_path.read_text(encoding="utf-8"))
    reasons = {}

    for record in table.get("absent", []):
        if set(record) != {"precision", "shape", "why"}:
            raise SystemExit(
                "check_class_surface: an `absent` record must carry exactly precision, shape and "
                f"why; this one carries {sorted(record)}")

        if not record["why"].strip():
            raise SystemExit(
                "check_class_surface: the absent class "
                f"({record['precision']}, {record['shape']}) states no reason")

        reasons[(record["precision"], record["shape"])] = record["why"]

    return reasons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="compare the tree against the table and report every class with no "
                             "entry; this is the form CI runs")
    parser.add_argument("--table", type=pathlib.Path, default=TABLE,
                        help="the shape table to read the `absent` reasons from; the tool's own "
                             "control runs it against a table that states one")
    args = parser.parse_args()

    text = visible(HEADER.read_text(encoding="utf-8", errors="replace"))
    precisions = enumerators(text, "Precision")
    shapes = enumerators(text, "Shape")

    # The device precisions are the device lane's classes, served from boys_cuda.hpp under their own
    # key; this check is the host surface, so they are not part of the cross it walks.
    host = [p for p in precisions if not p.endswith("Device")]
    served = set(CLASS.findall(code_only(text)))
    stated = absent_reasons(args.table)

    missing = []
    for precision in host:
        for shape in shapes:
            if (precision, shape) in served or (precision, shape) in stated:
                continue
            missing.append((precision, shape))

    print(f"check_class_surface: {len(host)} host precision(s) x {len(shapes)} shape(s) "
          f"= {len(host) * len(shapes)} class(es)")
    print(f"  served by an entry   {len(served & {(p, s) for p in host for s in shapes})}")
    print(f"  stated absent        {len(stated)}")

    if not missing:
        print("  every class in the cross is served or has a stated reason: PASS")
        return 0

    print(f"  with NO entry and NO stated reason: {len(missing)}")

    for precision, shape in missing:
        print(f"check_class_surface: ({precision}, {shape}) has no entry: no declaration in "
              f"{HEADER.name} defaults its policy to this class, and the table states no reason "
              f"for it. A class the library's own enumerations imply is either served or refused "
              f"with a proved impossibility, and work not done is not one.")

    return 1


if __name__ == "__main__":
    sys.exit(main())
