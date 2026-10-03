#!/usr/bin/env python
"""Hold the defaults table's classes to the entries that carry them.

WHY. This library keeps three hand-maintained lists that must agree, and until now nothing
checked that they do:

  1. the classes `BOYS_BUILD_DEFAULT_ROWS` carries, in include/boys/boys_build_defaults.hpp;
  2. the entries that accept a class default — the functions whose policy parameter defaults
     to `DefaultPolicy<Precision, Shape>`;
  3. the instantiations the library defines, in the extern lists and the out-of-line bodies.

Every build break of 2026-10-02 came from two of them disagreeing while each looked correct
from inside the list you were reading. A row for a class no entry reaches is a default nobody
can ask for, and it is invisible: the table reads complete, the header reads complete, and the
only symptom is that the class cannot be called. This script checks (1) against (2).

It checks one direction hard and the other one softly:
  * a row whose class no entry names is an ERROR — the row is unreachable;
  * an entry whose class no row names is an ERROR too, because a missing row is meant to be a
    compile error, so an entry naming an unrowed class would not build at all. Seeing one here
    means the two files were read differently and the reading is suspect.

The third list — whether the default instantiation can be LINKED — cannot be checked by reading
files, because `extern template` and out-of-line definitions are invisible to a parser. That half
is `tests/boys_defaults_link_test.cpp`, which calls every default at a multiplier no list holds.

Exit status is 0 when the lists agree and 1 with a per-class report when they do not.
"""

from __future__ import annotations

import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ROWS_FILE = os.path.join(REPO, "include", "boys", "boys_build_defaults.hpp")

# Every public header an entry can be declared in. The device surface declares its own entries
# and is swept too, so a device row added without an entry is caught the same way.
ENTRY_FILES = [
    os.path.join(REPO, "include", "boys", "boys.hpp"),
    os.path.join(REPO, "include", "boys", "boys_span.hpp"),
    os.path.join(REPO, "include", "boys", "boys_cuda.hpp"),
]

# A row of the table: X(kHost, kFp64, kAllOrders, ...) — the first three cells are the key.
ROW = re.compile(r"X\(\s*k(?:Host|Device)\s*,\s*(k[A-Za-z0-9]+)\s*,\s*(k[A-Za-z0-9]+)\s*,")

# An entry's default: the policy parameter spelled with a class key. The device's two extra axes
# are not part of an EvalPolicy, so a device entry spells the same `DefaultPolicy<...>` name with
# `Device::kDevice` as its third argument and is matched here as well.
ENTRY = re.compile(r"DefaultPolicy<\s*Precision::(k[A-Za-z0-9]+)\s*,\s*Shape::(k[A-Za-z0-9]+)"
                   r"(?:\s*,\s*Device::(k[A-Za-z0-9]+))?\s*>")


def read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def rows() -> dict[tuple[str, str], int]:
    """The table's classes, with the line each is written on."""
    found: dict[tuple[str, str], int] = {}
    for number, line in enumerate(read(ROWS_FILE).splitlines(), start=1):
        match = ROW.search(line)
        if match:
            found[(match.group(1), match.group(2))] = number
    return found


def entries() -> dict[tuple[str, str], list[tuple[str, int]]]:
    """The classes the entries name, with the file and line of each."""
    found: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for path in ENTRY_FILES:
        if not os.path.exists(path):
            continue
        # A device entry keys its class on Device::kDevice; both are the same class set here,
        # because what this checks is that a row and an entry name the same (precision, shape).
        for number, line in enumerate(read(path).splitlines(), start=1):
            for match in ENTRY.finditer(line):
                key = (match.group(1), match.group(2))
                found.setdefault(key, []).append((os.path.relpath(path, REPO), number))
    return found


def main() -> int:
    table = rows()
    named = entries()

    if not table:
        print(f"check_default_rows_have_entries: no rows found in "
              f"{os.path.relpath(ROWS_FILE, REPO)} - the reader is wrong, not the table")
        return 1

    unrowable = sorted(set(named) - set(table))
    unreachable = sorted(set(table) - set(named))

    for key in unreachable:
        line = table[key]
        print(f"check_default_rows_have_entries: the table carries a row for "
              f"({key[0]}, {key[1]}) at {os.path.relpath(ROWS_FILE, REPO)}:{line}, and NO entry "
              f"names that class: the row is a default nobody can ask for. Either add the entry "
              f"that carries it or take the row out - a class the library does not offer has no "
              f"default to state.")

    for key in unrowable:
        where = ", ".join(f"{path}:{line}" for path, line in named[key])
        print(f"check_default_rows_have_entries: an entry names the class ({key[0]}, {key[1]}) at "
              f"{where}, and the table carries no row for it. A class with no row is a compile "
              f"error by design, so either this reading is wrong or that entry cannot be built.")

    if unreachable or unrowable:
        print(f"\ncheck_default_rows_have_entries: {len(unreachable)} unreachable row(s), "
              f"{len(unrowable)} unrowed class(es), over {len(table)} row(s)")
        return 1

    print(f"check_default_rows_have_entries: all {len(table)} row(s) name a class an entry "
          f"carries, and every entry's class has a row")
    return 0


if __name__ == "__main__":
    sys.exit(main())
