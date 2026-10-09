#!/usr/bin/env python3
r"""Check that every surface a device row books is one the library declares *and* defines.

A row of the device option space names an entry, the entry's trailing doc names the surface
that answers it (`///< BoysCuda::Name at ...`), and four mechanisms consume that sentence
without any of them asking whether the surface is there. `tools/check_device_entry_bijection.py`
asks the first half of the question - is the name *declared* - and neither it nor any other
instrument in this tree asks the second: is it *defined*. So a row booking a surface the
library declares and never defines compiles nothing, links nothing, and is reported, counted
and probed as an option a caller can name.

That is not a hypothetical. The same shape has reached this tree four times, each time through
a different spelling of "declared but not there":

  * `BoysCuda::SingleF64Fast` - booked by a row, declared by no header;
  * forty-one further members of the same kind, found when the device surface was first
    linked whole;
  * thirty-one bf16 members, whose rows the device half of the class keyed and the class did
    not declare;
  * `BoysCuda::EachOrderBf16` and its siblings - declared, written as a template in a `.cpp`,
    and *instantiated nowhere*, so the definition emitted no symbol until one named it.

The first three are one defect - a booking with no declaration - and the bijection tool sees
them. The fourth is the one this check exists for, and it is why the defined set is read
separately rather than assumed from the declared one: a member whose body is a template in a
`.cpp` is a member whose definition reaches a caller only through an explicit instantiation
beside it, and "the body is in the file" is not the same fact as "a caller can reach it".

What is read, and from where:

  * booked - `include/boys/boys_cuda_options.hpp`, the `DeviceEntry` rows whose trailing doc
    names a surface, `///< BoysCuda::Name at ...` or `///< BoysDeviceName, ...`. The regex is
    the bijection tool's, so the two instruments cannot come to disagree about which rows
    exist;
  * rows - `src/boys_cuda.cpp`, the option table `kDeviceOptions`, to require that every row
    the library reports has an entry the enumeration books. A row whose entry books nothing is
    a row naming no surface at all, which the two other sets cannot see: they start from the
    enumeration;
  * declared - `include/boys/boys_cuda.hpp` (`static BoysStatus`) and
    `include/boys/boys_cuda_device.hpp` (`__device__ BoysDeviceStatus`), the class and the
    device half that answer the two spellings;
  * defined - `src/boys_cuda.cpp` for the launched surface, and the device header itself for
    the in-kernel one, which is header-only: there a declaration and a body are one statement,
    so a name is defined exactly when its declaration opens a brace rather than ending in one.

**A device name is an alias and not a second function.** `BoysDevice<X>` is the in-kernel
spelling of an entry, so it is compared against the device header's own declarations and the
launched names against the class's; the two namespaces are read side by side and never against
each other.

**The template rule is read from the class's own signature**, as the bijection tool reads its
own: a booked name whose declaration carries a template header is one whose body emits no
symbol until an instantiation names it, so an explicit instantiation of it is required beside
the body. Nothing is compiled and no symbol table is consulted - the two facts are read from
the text, at one revision, which is what makes the check runnable everywhere the suite is.

Usage:

    python3 tools/check_device_booked_surface.py --check
    python3 tools/check_device_booked_surface.py --options <file> --rows <file> \
        --header <file> --device-header <file> --source <file>   # negative control
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import NamedTuple

REPO = pathlib.Path(__file__).resolve().parent.parent
OPTIONS = REPO / "include" / "boys" / "boys_cuda_options.hpp"
HEADER = REPO / "include" / "boys" / "boys_cuda.hpp"
DEVICE_HEADER = REPO / "include" / "boys" / "boys_cuda_device.hpp"
ROWS = REPO / "src" / "boys_cuda.cpp"
# The translation unit the device class's own bodies are in. One file, because the class has one
# implementation file; `--source` reads a body side from elsewhere, which is how a body moved to a
# sibling file is shown to be found rather than reported missing.
SOURCES = [ROWS]

# A row of the enumeration that names the surface it books: `kName, ///< BoysCuda::Name at ...` or
# `kName, ///< BoysDeviceName, ...`. The first enumerator carries an explicit `= 0`, which this
# pattern allows; the bijection tool's pattern, so the two readings are one reading.
BOOKING = re.compile(r"^\s*(k\w+)\s*(?:=\s*\d+)?,\s*///<\s*(BoysCuda::\w+|BoysDevice\w+)")

# One row of the option table, and the entry it states.
ROW = re.compile(r"\{DeviceEntry::(\w+)\s*,")
ROW_TABLE = re.compile(r"constexpr DeviceOptionInfo kDeviceOptions\[\]\s*=\s*\{(.*?)\n\};",
                       re.DOTALL)

# The class's own members, and the device half's. The device spelling's return type may sit on a
# line of its own - the two halves of the lane's split style - so the type and the name are
# matched across whitespace rather than on one line, and neither pattern is anchored to a line.
LAUNCHED_DECLARATION = re.compile(r"static\s+BoysStatus\s+(\w+)\s*\(")
DEVICE_DECLARATION = re.compile(r"__device__\s+BoysDeviceStatus\s+(\w+)\s*\(")

# A body of one of the class's members, and one explicit instantiation of a member template. The
# qualified form is what makes these the class's definitions and not a free function's.
LAUNCHED_BODY = re.compile(r"BoysStatus\s+BoysCuda::(\w+)\s*\(")
INSTANTIATION = re.compile(r"^\s*template\s+BoysStatus\s+BoysCuda::(\w+)\s*<", re.MULTILINE)

# How far past a declaration to look for the brace or the semicolon that says which of the two it
# is. A member's head is a parameter list and, for a definition, one brace; the entry signatures
# are written one parameter per line, so the window is generous rather than a line count. Reading
# too far is harmless: the semicolon of a declaration ends the head before any brace can follow.
HEAD = 4000


class Booking(NamedTuple):
    """One enumerator, and the surface its own trailing doc names."""

    line: int
    entry: str
    surface: str

    @property
    def namespace(self) -> str:
        return "BoysCuda" if self.surface.startswith("BoysCuda::") else "device"


def code_only(text: str) -> str:
    """`text` with every comment line and trailing comment removed.

    A row that was commented out is not a row the library carries, and a body that was commented
    out is not a definition: prose must not be allowed to answer for either.
    """
    kept = []

    for line in text.splitlines():
        if line.lstrip().startswith("//"):
            continue

        kept.append(line.split("//", 1)[0])

    return "\n".join(kept)


def bookings_of(path: pathlib.Path) -> list[Booking]:
    """Every enumeration row that names a surface, in file order."""
    found = []

    for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        match = BOOKING.match(line)

        if match is not None:
            found.append(Booking(number, match.group(1), match.group(2)))

    return found


def rows_of(path: pathlib.Path) -> list[tuple[int, str]]:
    """Every row of the option table, as (line number, entry name)."""
    text = code_only(path.read_text(encoding="utf-8", errors="replace"))
    table = ROW_TABLE.search(text)

    if table is None:
        raise SystemExit(f"check_device_booked_surface: {path} declares no kDeviceOptions array: "
                         f"the option table this check walks is not in it")

    offset = text[:table.start(1)].count("\n") + 1
    found = []

    for match in ROW.finditer(table.group(1)):
        found.append((offset + table.group(1)[:match.start()].count("\n"), match.group(1)))

    return found


def defined_here(name: str, pattern: str, text: str) -> bool:
    """Whether `pattern` names `name` at a declaration that opens a body rather than ending in `;`."""
    for match in re.finditer(pattern, text):
        if match.group(1) != name:
            continue

        head = text[match.end():match.end() + HEAD]
        brace = head.find("{")
        semi = head.find(";")

        if brace >= 0 and (semi < 0 or brace < semi):
            return True

    return False


def declared_names(path: pathlib.Path, pattern: str, label: str) -> set[str]:
    """Every member `pattern` declares in `path`, and a hard stop when it declares none."""
    text = path.read_text(encoding="utf-8", errors="replace")
    found = {match.group(1) for match in re.finditer(pattern, text)}

    # The device spellings are matched across whitespace, so a stray mention of the return type in
    # prose would read as a declaration. The count is printed and an empty reading stops the run,
    # which is what keeps a pattern that has stopped matching from passing as a clean class.
    if not found:
        raise SystemExit(f"check_device_booked_surface: {path} declares no {label}, so nothing "
                         f"would be found declared and every booking would read as absent")

    return found


def templated_names(path: pathlib.Path) -> set[str]:
    """Every member of the class `path` declares with a template header on its own line.

    A template header sits on the line above its declaration, or on the line before one comment
    line, which is the layout the class uses.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    found: set[str] = set()
    last_template = -99

    for number, line in enumerate(lines):
        if line.strip().startswith("template"):
            last_template = number

        match = LAUNCHED_DECLARATION.search(line)

        if match is not None and number - last_template <= 3:
            found.add(match.group(1))

    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero on a surface a row books that is declared nowhere, or "
                             "declared and given no body a caller can reach; this is the form CI "
                             "would run")
    parser.add_argument("--options", type=pathlib.Path, default=OPTIONS,
                        help="the enumeration the bookings are read from")
    parser.add_argument("--header", type=pathlib.Path, default=HEADER,
                        help="the device class the launched bookings are read against")
    parser.add_argument("--device-header", type=pathlib.Path, default=DEVICE_HEADER,
                        help="the device half the in-kernel bookings are read against")
    parser.add_argument("--rows", type=pathlib.Path, default=ROWS,
                        help="the translation unit the option table is read from")
    parser.add_argument("--source", type=pathlib.Path, action="append", default=None,
                        help="a translation unit of the class's own bodies; repeatable")
    args = parser.parse_args()

    books = [args.options, args.header, args.device_header, args.rows]
    bodies = list(args.source) if args.source else list(SOURCES)
    missing = [path for path in books + bodies if not path.is_file()]

    if missing:
        print("check_device_booked_surface: missing " + ", ".join(str(path) for path in missing),
              file=sys.stderr)
        return 1

    booking_rows = bookings_of(args.options)
    option_rows = rows_of(args.rows)

    if not booking_rows or not option_rows:
        print(f"check_device_booked_surface: read {len(booking_rows)} booking(s) and "
              f"{len(option_rows)} row(s). A check that can pass by reading nothing is not a check",
              file=sys.stderr)
        return 1

    launched = declared_names(args.header, LAUNCHED_DECLARATION, "static BoysStatus member")
    device = declared_names(args.device_header, DEVICE_DECLARATION, "device-callable entry")
    templates = templated_names(args.header)

    body_text = {path: path.read_text(encoding="utf-8", errors="replace") for path in bodies}
    device_text = args.device_header.read_text(encoding="utf-8", errors="replace")
    instantiated = {name for path, text in body_text.items()
                    for name in INSTANTIATION.findall(text)}

    booked = sorted({booking.surface for booking in booking_rows})
    booked_rows = {booking.entry for booking in booking_rows}
    undeclared: list[Booking] = []
    undefined: list[tuple[Booking, str]] = []

    for booking in booking_rows:
        if booking.namespace == "BoysCuda":
            name = booking.surface.rsplit("::", 1)[-1]

            if name not in launched:
                undeclared.append(booking)
                continue

            has_body = any(defined_here(name, LAUNCHED_BODY, text) for text in body_text.values())

            if not has_body:
                undefined.append((booking, "no body defines it in "
                                           + ", ".join(str(path) for path in bodies)))
            elif name in templates and name not in instantiated:
                undefined.append((booking, "its body is a template in a translation unit and no "
                                           "explicit instantiation names it, so the definition "
                                           "emits no symbol"))
        else:
            if booking.surface not in device:
                undeclared.append(booking)
            elif not defined_here(booking.surface, DEVICE_DECLARATION, device_text):
                undefined.append((booking, "declared in "
                                           f"{args.device_header} with no body under it, so a "
                                           f"caller reaches a declaration and no code"))

    # A row whose entry no enumerator books names no surface at all, which the two sets above
    # cannot see: they start from the enumeration, and this row is not in it.
    unbooked = [(line, entry) for line, entry in option_rows if entry not in booked_rows]

    launched_booked = sum(1 for b in booking_rows
                          if b.namespace == "BoysCuda" and b.surface.rsplit("::", 1)[-1] in launched)
    device_booked = sum(1 for b in booking_rows
                        if b.namespace == "device" and b.surface in device)
    templated_booked = len(templates & {b.surface.rsplit("::", 1)[-1] for b in booking_rows})

    print(f"  enumeration rows booking a surface:  {len(booking_rows)}")
    print(f"  option rows of the library:          {len(option_rows)}")
    print(f"  distinct surfaces booked:            {len(booked)}")
    print(f"    declared by the class:             {launched_booked} booking(s)")
    print(f"    declared by the device half:       {device_booked} booking(s)")
    print(f"    member templates among them:       {templated_booked}")
    print(f"  declared members the class holds:    {len(launched)}")
    print(f"  declared entries the device half holds: {len(device)}")

    failures = len(undeclared) + len(undefined) + len(unbooked)

    if undeclared:
        print(f"\ncheck_device_booked_surface: {len(undeclared)} row(s) book a surface no "
              f"declaration names:", file=sys.stderr)

        for booking in undeclared:
            source = args.header if booking.namespace == "BoysCuda" else args.device_header
            print(f"  {booking.entry} (line {booking.line} of {args.options}) books "
                  f"{booking.surface}, and {source} declares no such "
                  f"{'member' if booking.namespace == 'BoysCuda' else 'entry'}: the option is "
                  f"advertised, reported and measured, and a caller who names it gets a compile "
                  f"error", file=sys.stderr)

    if undefined:
        print(f"\ncheck_device_booked_surface: {len(undefined)} row(s) book a surface that is "
              f"declared and given no body a caller can reach:", file=sys.stderr)

        for booking, why in undefined:
            print(f"  {booking.entry} (line {booking.line} of {args.options}) books "
                  f"{booking.surface}, which is declared and {why}: the row is a row of an option "
                  f"nothing can evaluate, and the linker is otherwise the first thing to say so",
                  file=sys.stderr)

    if unbooked:
        print(f"\ncheck_device_booked_surface: {len(unbooked)} row(s) of {args.rows} name an entry "
              f"the enumeration books no surface for:", file=sys.stderr)

        for line, entry in unbooked:
            print(f"  line {line}: DeviceEntry::{entry} is a row of the option table and no row of "
                  f"{args.options} books it, so the row names no surface this check can follow",
                  file=sys.stderr)

    if failures:
        return 1

    print(f"\ncheck_device_booked_surface: the {len(booking_rows)} booking(s) over "
          f"{len(booked)} distinct surface(s) all name a declared member with a body a caller can "
          f"reach, and the {len(option_rows)} row(s) of the option table all name a booked entry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
