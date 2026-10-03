#!/usr/bin/env python3
"""Check that every device option this build reports is an entry it declares, and the reverse.

The device surface is described by hand-written lists and nothing reconciles them: the
`DeviceEntry` enumeration in `boys_cuda_options.hpp`, the `BoysCuda` members in `boys_cuda.hpp`,
the device-callable declarations, and the probe's kernel arms. They
already disagreed once, in the worst possible direction: four rows of the enumeration named entries
that had never been declared, in any commit - and they
carried report rows with bounds and `built = true`, were launched by the measurement tool through a
twin's launcher, and were claimed by the accuracy gate, four separate mechanisms consumed the
description without one of them asking whether the entry was there.

This check is the bijection between the two lists a caller meets: what the enumeration says exists,
and what the class declares. Both directions are defects:

  * a row naming a member the class does not declare is an option the library advertises and cannot
    deliver - the published-but-absent defect above;
  * a member no row names is an entry the report books cannot describe, so nothing measures it and
    nothing counts it.

Usage:
  python3 tools/check_device_entry_bijection.py --check
  python3 tools/check_device_entry_bijection.py --options <file> --header <file>   # negative control
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

OPTIONS = pathlib.Path("include/boys/boys_cuda_options.hpp")
HEADER = pathlib.Path("include/boys/boys_cuda.hpp")

# An enum row's trailing doc names the entry it books: `///< BoysCuda::Name at ...`. The first
# enumerator carries an explicit `= 0`, which the row's own name does not need but the pattern must
# allow - missing it read the one entry every build has as unbooked.
ROW = re.compile(r"^\s*(k\w+)\s*(?:=\s*\d+)?,\s*///<\s*BoysCuda::(\w+)")

# Members of the device class that are not options and are booked by no row, each with the reason.
# They are declared here rather than filtered silently: a member that becomes infrastructure later
# has to be added with its reason, which is the point of the list.
NOT_AN_OPTION = {
    "DeviceTables": "it constructs the device tables; it is not an evaluation a report books",
    "InitializeTables": "it readies the lane before a call; it is not an evaluation",
}
# A member of the device class.
MEMBER = re.compile(r"^\s*static\s+BoysStatus\s+(\w+)\s*\(")


def rows_of(path: pathlib.Path) -> list[tuple[str, str]]:
    """Every `DeviceEntry` row that names an entry, as (row name, entry name)."""
    found: list[tuple[str, str]] = []

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ROW.match(line)

        if match is not None:
            found.append((match.group(1), match.group(2)))

    return found


def members_of(path: pathlib.Path) -> set[str]:
    """Every member the device class declares."""
    found: set[str] = set()

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = MEMBER.match(line)

        if match is not None:
            found.add(match.group(1))

    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit non-zero on a mismatch")
    parser.add_argument("--options", type=pathlib.Path, default=OPTIONS, help="the enumeration")
    parser.add_argument("--header", type=pathlib.Path, default=HEADER, help="the device class")
    args = parser.parse_args()

    if not args.options.is_file() or not args.header.is_file():
        print(f"check_device_entry_bijection: missing {args.options} or {args.header}", file=sys.stderr)
        return 1

    rows = rows_of(args.options)
    members = members_of(args.header)

    if not rows or not members:
        print(
            f"check_device_entry_bijection: read {len(rows)} row(s) and {len(members)} member(s). A "
            f"check that can pass by reading nothing is not a check",
            file=sys.stderr,
        )
        return 1

    named = {entry for _, entry in rows}
    declared = set(members)

    # A row books an entry, so that declaration must exist in its own right: an option advertised
    # under a name the class does not declare is as broken as one with no spelling at all.
    absent = sorted(named - members)

    # The other direction: every declared member answers to a row, or is exempted with a reason.
    unlisted = sorted(name for name in declared - named if name not in NOT_AN_OPTION)

    print(f"  enumeration rows naming an entry: {len(rows)}")
    print(f"  members the class declares:       {len(members)}")

    failures = 0

    if absent:
        failures += len(absent)
        print(f"\ncheck_device_entry_bijection: {len(absent)} row(s) name an entry the class does not "
              f"declare:", file=sys.stderr)

        for entry in absent:
            row = next(r for r, e in rows if e == entry)
            print(
                f"  {row} books BoysCuda::{entry}, and BoysCuda::{entry} is not declared in "
                f"{args.header}: the option is advertised, reported and measured, and a caller who "
                f"names it gets a compile error",
                file=sys.stderr,
            )

    if unlisted:
        failures += len(unlisted)
        print(f"\ncheck_device_entry_bijection: {len(unlisted)} declared member(s) no row names:",
              file=sys.stderr)

        for entry in unlisted:
            print(
                f"  BoysCuda::{entry} is declared in {args.header} and no row of {args.options} "
                f"books it, so the report cannot describe it and no book counts it",
                file=sys.stderr,
            )

    if failures:
        return 1

    print(f"\ncheck_device_entry_bijection: the {len(named)} booked entr(ies) and the "
          f"{len(declared)} declared member(s) are the same set")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
