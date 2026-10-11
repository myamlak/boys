#!/usr/bin/env python3
"""The stamp a recorded run or report carries, and how a checker reads it back.

A record kept as a file - the accuracy gate's run under `tests/data`, the option probe's report
beside it - is one build's output on one host, and a document that quotes a figure from it is
quoting that build. The reader of the document needs one answer: is the tree in front of them
the tree the record describes? Two cells of the record's stamp carry it.

    version <v>        the release, `MAJOR.MINOR.PATCH`, as `include/boys/version.hpp` states
                       it - the same string `boys::VersionString()` returns
    inputs <cells>     one digest per group of the files the program reads from, written
                       `group=digest` and separated by commas

**The digest is the check and the version is what a human reads.** The version cell is what a
reader places a record by, and it is bumped by hand; a source file edited under a header nobody
bumped leaves the cell naming a content the tree has left, so the cell alone is not what a
checker may decide on. The digest is the content itself, recomputed at check time and compared
group by group.

Content is also the identity that survives this repository's merge. A pull request is squash
merged, so a commit id stamped into a tracked artifact is in no clone's history afterwards, and
a checker resolving one refuses a record for a reason about the merge rather than about the
tree. A squash carries the same bytes, so a digest survives the merge that a commit id does not.

**The bytes are read with CRLF folded to LF, and the reason is a false finding.** The
repository is LF and `.gitattributes` says so, while an editor on one host can write CRLF into
the working tree; git normalizes such a file on its way into the index, so `git status` reports
that tree clean and a digest taken raw disagrees with the same content read where it was checked
out LF. Folding the ending is what makes one content read the same on every host, which is the
identity this stamp claims to carry, and the five carriers of CRLF in this working tree are
what it was measured on.

A digest is the sha256 of a group's files, each written as its repository-relative path, a
newline, its own sha256 in hex and a newline, taken in sorted path order; the first
`DIGEST_CHARS` characters of that hash are the cell's value. A group holds a file as itself and
a directory as every regular file under it, read recursively.

Usage:
    python3 tools/record_stamp.py --inputs probe             # the cell a probe build stamps in
    python3 tools/record_stamp.py --inputs gate --root <dir> # the cell a gate build stamps in
    python3 tools/record_stamp.py --version                  # the version the tree states

Exit status: 0 when the cell or the version was read, and 1 when a path a record's own input
list names is not in the tree at `--root`, or when the version header states none.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

# Where the tree states its own version. `CMakeLists.txt` writes the number into the project and
# the header repeats it, a configure asserts the two agree, and this reads the header because the
# header is what a program is compiled against and therefore what a record prints.
VERSION_HEADER = "include/boys/version.hpp"
VERSION = re.compile(r'kVersionString\s*=\s*"(?P<version>[^"]+)"')

# What a digest keeps of a file's sha256. Sixteen hexadecimal characters is a cell a line can
# carry four of; a collision needs two trees that agree over 64 bits of a construction which has
# already hashed every file.
DIGEST_CHARS = 16

# The cell a build states when it could not compute one: a configure with no interpreter to run
# this tool, or a build whose own command line passed no digest. A checker refuses a record
# carrying it rather than reading it as a digest that matches nothing.
UNKNOWN = "unknown"

# The files each record is a record of, group by group. Every file a program reads belongs to a
# group, and a group is what the stamp's cell names and what a finding can point at.
#
# `cmake` is in both lists and is gated by neither, for a reason that is a limit of reading: a
# configure decides the flags a target compiles under and it decides which sources are in which
# target, and a file's digest does not say which of the two a change was. A build-description edit
# that added a test to another target would fail a check that gated it, and one that moved an
# arithmetic option would pass a check that ignored it. The cell carries its content, so a reader
# of the stamp sees that the file moved; the finding waits for the reader who has the diff.
INPUTS: dict[str, dict[str, tuple[str, ...]]] = {
    "probe": {
        "src": ("src",),
        "include": ("include",),
        "driver": ("benchmarks/boys_option_probe.cpp",),
        "cmake": ("CMakeLists.txt",),
    },
    "gate": {
        "src": ("src",),
        "include": ("include",),
        "gate": ("tests/boys_accuracy_gate.cpp", "tests/boys_gate_reference.hpp"),
        "grid": ("tests/data/boys_accuracy_gate_reference.csv",),
        "cmake": ("CMakeLists.txt",),
    },
}

# A group no finding is raised for, whatever its digest says.
NOT_GATED = frozenset({"cmake"})


class StampError(Exception):
    """A stamp that could not be read from the tree it was asked about."""


def version(root: pathlib.Path = REPO) -> str:
    """The version the tree states: `kVersionString` in `include/boys/version.hpp`."""
    header = root / VERSION_HEADER
    try:
        text = header.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise StampError(f"{header} could not be read: {error}") from error
    found = VERSION.search(text)
    if found is None:
        raise StampError(f"{header} states no kVersionString")
    return found.group("version")


def files(root: pathlib.Path, paths: tuple[str, ...]) -> list[str]:
    """The repository-relative paths of one group, sorted: a file as itself, a directory walked."""
    found: list[str] = []
    for entry in paths:
        target = root / entry
        if target.is_file():
            found.append(entry)
        elif target.is_dir():
            for child in target.rglob("*"):
                if child.is_file():
                    found.append(child.relative_to(root).as_posix())
        else:
            raise StampError(f"{entry} is not in the tree at {root}")
    return sorted(set(found))


def digest(root: pathlib.Path, paths: tuple[str, ...]) -> str:
    """One group's digest: its files, their content, in sorted path order."""
    accumulator = hashlib.sha256()
    for relative in files(root, paths):
        content = (root / relative).read_bytes().replace(b"\r\n", b"\n")
        accumulator.update(relative.encode("utf-8"))
        accumulator.update(b"\n")
        accumulator.update(hashlib.sha256(content).hexdigest().encode("ascii"))
        accumulator.update(b"\n")
    return accumulator.hexdigest()[:DIGEST_CHARS]


def digests(root: pathlib.Path, record: str) -> dict[str, str]:
    """Every group of one record's input list, by name, in the order that list states."""
    return {name: digest(root, paths) for name, paths in INPUTS[record].items()}


def gated(record: str) -> tuple[str, ...]:
    """The groups of a record a finding may be raised for."""
    return tuple(name for name in INPUTS[record] if name not in NOT_GATED)


def cell(values: dict[str, str]) -> str:
    """A cell's text: `group=digest` pairs, comma-separated, in the order given."""
    return ",".join(f"{name}={value}" for name, value in values.items())


def parse_cell(text: str) -> dict[str, str]:
    """The pairs a cell states, as text.

    A token without a `=` is read as a group with an empty value rather than dropped, so a cell
    that names a group and no digest is found short by the caller instead of passing as absent.
    """
    values: dict[str, str] = {}
    for token in text.replace(" ", ",").split(","):
        if token:
            name, _, value = token.partition("=")
            values[name] = value
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", choices=sorted(INPUTS),
                        help="print the stamp cell for this record's own input list, read at "
                             "the tree --root names")
    parser.add_argument("--version", action="store_true",
                        help="print the version the tree at --root states")
    parser.add_argument("--root", type=pathlib.Path, default=REPO,
                        help="the tree to read (default: the repository this tool is in)")
    args = parser.parse_args()

    if not args.inputs and not args.version:
        parser.error("ask for --inputs NAME or --version; a run that reads nothing prints nothing")

    try:
        if args.version:
            print(version(args.root))
        if args.inputs:
            print(cell(digests(args.root, args.inputs)))
    except (OSError, StampError) as error:
        print(f"record_stamp: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
