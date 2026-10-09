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

The two lists are of different sizes, because one option may be reached under more than one
spelling. A member named `<X>WithPolicy` whose `<X>` is an entry a row books, and whose declaration
takes an `EvalPolicyLike` policy, is that entry reached by naming a policy while the call site
compiles instead of by the entry's name: the same option, a second spelling, an entry the row
already books, so it books nothing of its own. Both halves of that rule are read, because each
excludes a case the other would wrongly accept - a `FooWithPolicy` whose declaration takes no policy
is a new entry that happens to be named after `Foo`, and a `<X>WithPolicy` whose `<X>` no row books
names an option this table does not describe; and a rule matching the suffix alone would take either
one for a spelling and stop checking it. Neither is exempted here. Both are reported.

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

# The suffix a policy-dispatch spelling carries, and the constraint its declaration carries with it.
# The suffix alone is a name; the constraint is what makes the call a dispatch on a policy.
POLICY_SUFFIX = "WithPolicy"
POLICY_CONSTRAINT = "EvalPolicyLike"


def rows_of(path: pathlib.Path) -> list[tuple[str, str]]:
    """Every `DeviceEntry` row that names an entry, as (row name, entry name)."""
    found: list[tuple[str, str]] = []

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ROW.match(line)

        if match is not None:
            found.append((match.group(1), match.group(2)))

    return found


def header_is_complete(header: str) -> bool:
    """True when every `<` a template header opened has been closed."""
    return header.count("<") > 0 and header.count("<") == header.count(">")


def declarations_of(path: pathlib.Path) -> dict[str, str]:
    """Every member the device class declares, mapped to the template header on its declaration.

    The header is read because it is the only evidence in this file that a member is a policy
    dispatch rather than an option of its own, and a name is not evidence: a new entry is free to
    end in `WithPolicy` as well. A member declared with no template header maps to "".

    A header may wrap over more than one line, and the class's declarations are the only thing it
    belongs to, so a header ends at the first member declaration after it. A member is matched
    before the header is extended: a header whose `<` never closes must not swallow the declaration
    under it, which would read the class as declaring fewer members than it does.
    """
    found: dict[str, str] = {}
    header: str | None = None

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        match = MEMBER.match(line)

        if match is not None:
            found[match.group(1)] = header or ""
            header = None
        elif stripped.startswith("template"):
            header = stripped
        elif header is not None and not header_is_complete(header):
            header = f"{header} {stripped}"
        elif header is not None and not (
            stripped == "" or stripped.startswith(("//", "*", "/*", "*/"))
        ):
            # A line of code that is not this header's declaration: the header belongs to whatever
            # it was written for, and this member is not it.
            header = None

    return found


def policy_spellings(
    declared: set[str], booked: set[str], headers: dict[str, str]
) -> dict[str, str]:
    """Each member that is the policy-dispatch spelling of a booked entry, and the entry it spells.

    Two facts are required, the name and the declaration. `<X>` must be an entry a row already
    books, so the member reaches an option this table describes; and the declaration must take an
    `EvalPolicyLike` policy, so the member is a call dispatched on a policy rather than a new entry
    named after one.
    """
    found: dict[str, str] = {}

    for name in sorted(declared):
        if not name.endswith(POLICY_SUFFIX):
            continue

        base = name[: -len(POLICY_SUFFIX)]

        if base in booked and POLICY_CONSTRAINT in headers.get(name, ""):
            found[name] = base

    return found


def refusal_of(name: str, booked: set[str], headers: dict[str, str]) -> str | None:
    """Why a `*WithPolicy` member is no spelling of a booked entry, or None when it is one.

    An unlisted member carrying the suffix is the case a reader has to decide about, so the report
    says which half of the rule it failed: the option it names is one no row books, or the call is
    not a policy dispatch at all.
    """
    if not name.endswith(POLICY_SUFFIX):
        return None

    base = name[: -len(POLICY_SUFFIX)]

    if base not in booked:
        return (
            f"its base name {base} is an entry no row books, so it names an option this table does "
            f"not describe"
        )

    return (
        f"its declaration takes no {POLICY_CONSTRAINT} policy, so it is an entry of its own that "
        f"happens to be named after {base}, not a policy spelling of it"
    )


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
    declarations = declarations_of(args.header)
    members = set(declarations)

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

    # The other direction: every declared member books itself by answering to a row, or is the
    # policy spelling of a member that does, or is named here as no option. The three are cut from
    # what is left in that order, so they partition the class: a member two of them claim would
    # make the sentence the run ends on untrue while the check stayed green.
    booked = declared & named
    spellings = policy_spellings(declared - booked, named, declarations)
    exempt = sorted((declared - booked - set(spellings)) & set(NOT_AN_OPTION))
    unlisted = sorted(declared - booked - set(spellings) - set(exempt))

    # The list of members that are no option is read back against the class like every other
    # hand-written list here: a name on it that the class does not declare describes nothing, and
    # its reason then hides whichever member took its place.
    stale = sorted(set(NOT_AN_OPTION) - declared)

    print(f"  enumeration rows naming an entry: {len(rows)}")
    print(f"  members the class declares:       {len(members)}")
    print(f"    booked by a row:                {len(booked)}")
    print(f"    policy spellings of them:       {len(spellings)}")
    print(f"    no option:                      {len(exempt)}")

    if spellings:
        print(f"\n  reachable by naming a policy, booking nothing of their own:")
        for name in sorted(spellings):
            print(f"    BoysCuda::{name} -> BoysCuda::{spellings[name]}")

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
        print(f"\ncheck_device_entry_bijection: {len(unlisted)} declared member(s) no row names and "
              f"no rule books:", file=sys.stderr)

        for entry in unlisted:
            refusal = refusal_of(entry, named, declarations)
            because = f" {refusal}." if refusal else ""

            print(
                f"  BoysCuda::{entry} is declared in {args.header} and no row of {args.options} "
                f"books it, so the report cannot describe it and no book counts it.{because}",
                file=sys.stderr,
            )

    if stale:
        failures += len(stale)
        print(f"\ncheck_device_entry_bijection: {len(stale)} name(s) held as no option the class "
              f"does not declare:", file=sys.stderr)

        for entry in stale:
            print(
                f"  BoysCuda::{entry} is held as no option ({NOT_AN_OPTION[entry]}), and no such "
                f"member is declared in {args.header}: the list describes a class this is not, so "
                f"a member it was written for is either renamed or gone and the reason hides which",
                file=sys.stderr,
            )

    if failures:
        return 1

    print(f"\ncheck_device_entry_bijection: the {len(booked)} booked entr(ies), the "
          f"{len(spellings)} policy spelling(s) of them and the {len(exempt)} member(s) that are no "
          f"option are the {len(declared)} member(s) the class declares")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
