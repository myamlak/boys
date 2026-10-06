#!/usr/bin/env python3
"""Put an emitted seam's rows into the committed seam, and prove they arrived whole.

`boys-option-probe --emit-defaults` and `boys-device-probe --emit-defaults` write a
header carrying the rows a run measured. The committed seam carries the rows a build
resolves to. This copies the rows of one into the other.

It copies **rows and nothing else**. The prose around them is written by a person and
is the part a splice gets wrong: the last one replaced the rows and left the previous
run's sentence standing, so the file named a run it had not come from and nothing
noticed until the emitted file was read beside it. A tool that rewrites the whole file
would have made the same mistake faster, so this one refuses to touch a line that is
not a row.

What it checks, and refuses on:

* every row the emitted file carries it also carries a marker for, or neither;
* the rows it is about to write are byte-identical to the emitted file's, read back
  after the write rather than assumed from the write;
* the number of rows replaced is the number the emitted file states.

Usage:

    python tools/splice_default_rows.py --emitted <file> --seam <file> [--device-only]
    python tools/splice_default_rows.py --emitted <file> --seam <file> --check

Exit status: 0 when the seam carries the emitted rows; 1 when it does not and was not
asked to change; 2 when the two files cannot be reconciled - a row without a marker,
a count that disagrees with the emission's own statement.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# A row and the marker comment over it. A row is `X(...)` and nothing else; the marker
# is the `/* ... */` run immediately above it, which is where a run states how the row
# was reached. They are read as a pair because a row without its marker is a row whose
# provenance a reader cannot check.
# A row's text once its continuation backslashes are taken off. A row is `X(...)` and
# nothing else, so a line of prose that happens to start with `X(` is not one.
ROW = re.compile(r"^\s*X\((?:[^()]|\([^()]*\))*\)\s*$")
MARKER = re.compile(r"^\s*/\*.*\*/\s*$")

DEVICE = "X(kDevice"

# Every line of the rows block ends with a backslash, because the whole list is one
# macro definition: the continuation is the macro's, not the row's, so it says nothing
# about where a row begins or ends. What does is the row's own shape - it opens with
# `X(` and closes with the `)` that balances it - read on each line's content.
CONTINUED = "\\"
ROW_OPENS = re.compile(r"^\s*X\(")
ROW_CLOSES = re.compile(r"\)\s*$")


def content(line: str) -> str:
    """A physical line without the macro's continuation backslash."""
    stripped = line.rstrip()
    if stripped.endswith(CONTINUED):
        stripped = stripped[:-1]
    return stripped.rstrip()


def elements(text: str) -> list[tuple[str, list[str], list[str]]]:
    """The text as elements: a row with its marker, or lines that are neither.

    Each element is `(kind, marker_lines, body_lines)`. A marker is the run of
    `/* ... */` lines immediately above a row; a line that is neither a marker nor
    part of a row ends the run, so a paragraph of prose above a block is not carried
    into the next row's marker.
    """
    lines = text.splitlines()
    found: list[tuple[str, list[str], list[str]]] = []
    pending: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]

        if ROW_OPENS.match(content(line)):
            body = [line.rstrip()]
            index += 1
            while not ROW_CLOSES.search(content(body[-1])) and index < len(lines):
                body.append(lines[index].rstrip())
                index += 1
            found.append(("row", pending[-12:], body))
            pending = []
            continue

        if MARKER.match(content(line)):
            pending.append(line.rstrip())
            index += 1
            continue

        if pending:
            found.append(("other", [], pending))
            pending = []

        found.append(("other", [], [line.rstrip()]))
        index += 1

    if pending:
        found.append(("other", [], pending))

    return found


def read_rows(text: str) -> list[tuple[list[str], list[str]]]:
    return [(marker, body) for kind, marker, body in elements(text) if kind == "row"]


def is_device(row: tuple[list[str], list[str]]) -> bool:
    return any(DEVICE in line for line in row[1])


def splice(seam_text: str, replacement: list[tuple[list[str], list[str]]], device_only: bool) -> str:
    """The seam with its rows replaced, marker and row together."""
    out: list[str] = []
    taken = 0

    for kind, marker, body in elements(seam_text):
        if kind != "row":
            out.extend(marker)
            out.extend(body)
            continue

        if device_only and not is_device(("", body)):
            out.extend(marker)
            out.extend(body)
            continue

        if taken >= len(replacement):
            raise SystemExit(
                "splice_default_rows: the seam carries more rows than the emission "
                "does; the two cannot be reconciled by position"
            )

        new_marker, new_body = replacement[taken]
        out.extend(new_marker)
        out.extend(new_body)
        taken += 1

    if taken != len(replacement):
        raise SystemExit(
            f"splice_default_rows: the seam carries {taken} row(s) to replace and the "
            f"emission states {len(replacement)} - a row the seam does not carry is a "
            f"class the build resolves another way"
        )

    return "\n".join(out) + "\n"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--emitted", required=True, help="the file a run emitted")
    parser.add_argument("--seam", required=True, help="the committed seam to write")
    parser.add_argument("--device-only", action="store_true",
                        help="replace the kDevice rows only, leaving the host rows alone")
    parser.add_argument("--check", action="store_true",
                        help="report whether they agree; change nothing")
    options = parser.parse_args(argv)

    emitted_path = Path(options.emitted)
    seam_path = Path(options.seam)
    emitted_text = emitted_path.read_text(encoding="utf-8")
    seam_text = seam_path.read_text(encoding="utf-8")

    emitted = read_rows(emitted_text)
    replacement = [row for row in emitted if is_device(row)] if options.device_only else emitted

    if not replacement:
        print(f"splice_default_rows: {emitted_path} states no row to splice", file=sys.stderr)
        return 2

    for marker, _body in replacement:
        if not marker:
            print(
                "splice_default_rows: a row of the emission carries no marker, so how it "
                "was reached cannot be read beside it",
                file=sys.stderr,
            )
            return 2

    spliced = splice(seam_text, replacement, options.device_only)

    # Read back what would be written and compare it against the emission, rather than
    # trusting the construction above.
    arrived = read_rows(spliced)
    arrived_rows = [row for row in arrived if is_device(row)] if options.device_only else arrived

    if arrived_rows != replacement:
        print("splice_default_rows: the rows that arrive are not the rows the emission "
              "states - refusing to write", file=sys.stderr)
        return 2

    if options.check:
        if spliced == seam_text:
            print(f"splice_default_rows: {seam_path} carries the emission's "
                  f"{len(replacement)} row(s) already")
            return 0
        print(
            f"splice_default_rows: {seam_path} does NOT carry the emission's "
            f"{len(replacement)} row(s); {len(replacement)} differ",
            file=sys.stderr,
        )
        return 1

    seam_path.write_text(spliced, encoding="utf-8")
    print(f"splice_default_rows: {len(replacement)} row(s) of {emitted_path.name} written "
          f"into {seam_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
