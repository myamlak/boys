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
  * a row whose class nothing reaches - no entry's default, and no unit asking for the class by
    name (ASK_FILES) - is an ERROR: the row is unreachable;
  * an entry or an asking unit whose class no row names is an ERROR too, because a missing row is
    meant to be a compile error, so a name for an unrowed class would not build at all. Seeing one
    here means the two files were read differently and the reading is suspect.

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

# The units that ask for a class BY NAME rather than carrying it as an entry's default. A class is
# reached one of two ways, and the two halves of the table use the two ways:
#
#   * a host class is reached by the entries whose policy parameter defaults to it, which is what
#     the header sweep above reads;
#   * a device class is reached by the default-policy name a unit asks for it with. The device
#     surface's entries take a division form and a region-B member and no policy parameter, so no
#     entry declares the row - and the row is not therefore unreachable: `DefaultPolicy<Precision,
#     Shape, Device>` is the public name for a class's default, and a unit that names one of the
#     nine asks the table for it. tests/boys_build_defaults_test.cpp asks for all nine and is what
#     makes a missing device row a build error (DefaultPolicyFor asserts on it), so a device row
#     nothing asks for is a row this reports as unreachable like any other.
ASK_FILES = [
    os.path.join(REPO, "tests", "boys_build_defaults_test.cpp"),
]

# A row of the table: X(kHost, kFp64, kAllOrders, ...) — the first three cells are the key.
ROW = re.compile(r"X\(\s*k(?:Host|Device)\s*,\s*(k[A-Za-z0-9]+)\s*,\s*(k[A-Za-z0-9]+)\s*,")

# An entry's default: the policy parameter spelled with a class key. The device's two extra axes
# are not part of an EvalPolicy, so a device entry spells the same `DefaultPolicy<...>` name with
# `Device::kDevice` as its third argument and is matched here as well. The axes may be spelled
# with a namespace prefix - a unit outside the library's namespace writes `boys::Precision::kFp64`
# - so a name is matched with whatever qualification it carries.
ENTRY = re.compile(r"DefaultPolicy<\s*(?:[A-Za-z_][A-Za-z0-9_]*::)*Precision::(k[A-Za-z0-9]+)\s*,"
                   r"\s*(?:[A-Za-z_][A-Za-z0-9_]*::)*Shape::(k[A-Za-z0-9]+)"
                   r"(?:\s*,\s*(?:[A-Za-z_][A-Za-z0-9_]*::)*Device::(k[A-Za-z0-9]+))?\s*>")


def read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def code(path: str) -> str:
    """The file's text with its comments blanked out, line structure kept.

    A name is read out of the code and not out of what the code says about itself. This
    library's headers explain the shape of an entry's default in prose, and prose spells
    the same `DefaultPolicy<Precision::k..., Shape::k...>` an entry does: boys.hpp's own
    comment on the composed table writes the placeholders `kX` and `kY` in exactly that
    form, and a reader that does not skip comments books a class called (kX, kY) that no
    row could ever carry. Blanking rather than dropping keeps every line number the line
    number it was, which is what the reports below print.
    """
    text = read(path)
    out: list[str] = []
    index = 0
    while index < len(text):
        pair = text[index:index + 2]
        if pair == "//":
            end = text.find("\n", index)
            end = len(text) if end < 0 else end
            out.append(" " * (end - index))
            index = end
        elif pair == "/*":
            end = text.find("*/", index + 2)
            end = len(text) if end < 0 else end + 2
            out.append("".join("\n" if character == "\n" else " " for character in text[index:end]))
            index = end
        else:
            out.append(text[index])
            index += 1
    return "".join(out)


def code_lines(path: str) -> list[tuple[int, str]]:
    """The file's lines with comments blanked, as (line number, text).

    The line-at-a-time reading of `code` above, for the caller that reads a thing written on
    one line: a row is one line of the table, and a class's columns are read off it. An entry's
    default is not - it is asked for across as many lines as the width allows - so the reader
    that looks for one walks the whole text and this one is not offered to it. Deriving both
    from one blanking is what keeps the two callers from disagreeing about what a comment is.

    A class named in prose is not a declaration. The header's own commentary spells the row
    format out with the placeholder `DefaultPolicy<Precision::kX, Shape::kY>`, which is the same
    text a real entry writes, so a check that scans raw lines reads that sentence as a class
    named (kX, kY) and reports an entry no row can carry. The line numbers kept here are the
    file's own, so a report still names a line a reader can open.
    """
    return list(enumerate(code(path).splitlines(), start=1))


def rows() -> dict[tuple[str, str], int]:
    """The table's classes, with the line each is written on."""
    found: dict[tuple[str, str], int] = {}
    for number, line in code_lines(ROWS_FILE):
        match = ROW.search(line)
        if match:
            found[(match.group(1), match.group(2))] = number
    return found


def entries() -> dict[tuple[str, str], list[tuple[str, int]]]:
    """The classes the entries and the asking units name, with the file and line of each.

    A name is matched over the whole file rather than line by line, because a class's name
    is written the way the line length allows: the device classes are asked for as
    `DefaultPolicy<...Precision::kFp64Device, Shape::kSingle,\\n ... Device::kDevice>`,
    and a reader that stops at the newline sees a `DefaultPolicy<` that never closes and
    books the class the ask reaches as a row nobody asks for.
    """
    found: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for path in ENTRY_FILES + ASK_FILES:
        if not os.path.exists(path):
            continue
        # A device entry keys its class on Device::kDevice; both are the same class set here,
        # because what this checks is that a row and an entry name the same (precision, shape).
        # Walked over the blanked text and not line by line: an entry's default is written the
        # way the line width allows, so a class asked for across a line break has to be read
        # across it. The blanking is `code` above, which is also what `code_lines` reads, so
        # what counts as a declaration is one reading in both places.
        text = code(path)
        for match in ENTRY.finditer(text):
            key = (match.group(1), match.group(2))
            number = text.count("\n", 0, match.start()) + 1
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
              f"and no asking unit names that class: the row is a default nobody can ask for. "
              f"Either add the entry that carries it - or, on the device half, the unit that asks "
              f"for it by name - or take the row out: a class the library does not offer has no "
              f"default to state.")

    for key in unrowable:
        where = ", ".join(f"{path}:{line}" for path, line in named[key])
        print(f"check_default_rows_have_entries: an entry or an asking unit names the class "
              f"({key[0]}, {key[1]}) at {where}, and the table carries no row for it. A class with "
              f"no row is a compile error by design, so either this reading is wrong or that name "
              f"cannot be built.")

    if unreachable or unrowable:
        print(f"\ncheck_default_rows_have_entries: {len(unreachable)} unreachable row(s), "
              f"{len(unrowable)} unrowed class(es), over {len(table)} row(s)")
        return 1

    print(f"check_default_rows_have_entries: all {len(table)} row(s) name a class an entry or an "
          f"asking unit reaches, and every name's class has a row")
    return 0


if __name__ == "__main__":
    sys.exit(main())
