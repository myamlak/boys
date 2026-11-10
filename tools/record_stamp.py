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
The accuracy gate's record was the last carrier of a commit id beside its content; that line
was retired in 2026-10, and the record's identity is now its version and these digests.

**A version is metadata, and a version-only bump leaves a record true.** `include/boys/version.hpp`
holds `kVersionString` and the three numbers and nothing a run computes with, so a release that
moves that header and no other input is the same code with a later name. A record naming the
earlier release still describes the tree, and `differs_only_in_version_header` is what proves it:
the record names version X, so tag `vX` is the tree it was taken from, and the difference is the
header alone only where git shows the header to be the one gated input that moved and the
record's digests to be the tag's own. With no repository or no tag there is nothing to prove it
with, and the difference is refused rather than assumed harmless.

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
import subprocess
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

# The prefix a release tag carries. A record names version `5.3.1`, and tag `v5.3.1` is the tree
# that release published, so the tag is what a record's own version line can be held to.
TAG_PREFIX = "v"

# The files a record's input groups name that carry release metadata rather than an arithmetic
# input: `version.hpp` states the version string and the three numbers, and no figure a run
# prints is computed from them. A record whose own release moved only here describes this code.
VERSION_METADATA = frozenset({VERSION_HEADER})


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


def digest(root: pathlib.Path, paths: tuple[str, ...],
           content: dict[str, bytes] | None = None) -> str:
    """One group's digest: its files, their content, in sorted path order.

    A file named in `content` is hashed from those bytes instead of from the tree, which is how a
    group is read as a release tag holds it: the tag's version header is read from git, and every
    other file of the group is read from the tree a diff has certified unchanged since the tag.
    """
    overrides = content or {}
    accumulator = hashlib.sha256()
    for relative in files(root, paths):
        data = overrides.get(relative)
        if data is None:
            data = (root / relative).read_bytes()
        data = data.replace(b"\r\n", b"\n")
        accumulator.update(relative.encode("utf-8"))
        accumulator.update(b"\n")
        accumulator.update(hashlib.sha256(data).hexdigest().encode("ascii"))
        accumulator.update(b"\n")
    return accumulator.hexdigest()[:DIGEST_CHARS]


def digests(root: pathlib.Path, record: str) -> dict[str, str]:
    """Every group of one record's input list, by name, in the order that list states."""
    return {name: digest(root, paths) for name, paths in INPUTS[record].items()}


def gated(record: str) -> tuple[str, ...]:
    """The groups of a record a finding may be raised for."""
    return tuple(name for name in INPUTS[record] if name not in NOT_GATED)


def metadata_groups(record: str) -> tuple[str, ...]:
    """The groups of a record whose file list carries version metadata."""
    return tuple(
        name for name, paths in INPUTS[record].items()
        if any(metadata == entry.rstrip("/")
               or metadata.startswith(entry.rstrip("/") + "/")
               for entry in paths for metadata in VERSION_METADATA)
    )


def _git(root: pathlib.Path, *args: str) -> tuple[int, str, str]:
    """git at `root`, as (exit status, stdout, stderr). stdout is not trimmed."""
    try:
        proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    except OSError as error:
        return 127, "", f"git could not be run: {error}"
    return proc.returncode, proc.stdout, proc.stderr.strip()


def _git_blob(root: pathlib.Path, name: str) -> tuple[int, bytes, str]:
    """One git object's bytes, as the object store holds them."""
    try:
        proc = subprocess.run(["git", "-C", str(root), "show", name], capture_output=True)
    except OSError as error:
        return 127, b"", f"git could not be run: {error}"
    return proc.returncode, proc.stdout, proc.stderr.decode("utf-8", "replace").strip()


def differs_only_in_version_header(root: pathlib.Path, named: str, record: str,
                                   recorded: dict[str, str]) -> tuple[bool, str]:
    """Whether the tree at `root` is tag `v<named>`'s code with the version header alone moved.

    A record names a release, and the tag `v` plus that version is the tree it was taken from.
    A later tree that has moved only in `include/boys/version.hpp` carries the same arithmetic
    with a later release string, so the record still describes it.

    The answer is proved, not assumed. The tag must resolve; the diff of the tree against it over
    the record's gated groups must name the version header and nothing else, untracked files
    included; and the record's own digests must be the tag's, with the header's content at the tag
    hashed in place of the tree's. A group that does not match is a record of another tree and not
    a version difference.

    Returns (True, "") when the difference is the version header alone, and (False, why) with a
    sentence a caller's finding can carry otherwise. A missing repository or tag is a failure and
    not a pass: without the tag there is no tree to hold the record to, and the difference is
    refused rather than assumed harmless. `cmake` is not read, for the reason the checkers gate
    the other groups and not it: a digest does not say which of a flag and a target list a change
    to it was.
    """
    status, _, error = _git(root, "rev-parse", "--git-dir")
    if status != 0:
        return False, (
            f"This is not a git repository that can be read ({error or f'git exited {status}'}), "
            f"so the difference between version {named} and this tree cannot be verified as the "
            f"version header alone, and this check fails closed")

    tag = TAG_PREFIX + named
    status, _, error = _git(root, "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}")
    if status != 0:
        return False, (
            f"The record's own tag `{tag}` is not in this repository ({error or 'no such tag'}), "
            f"so the difference between version {named} and this tree cannot be verified as the "
            f"version header alone, and this check fails closed")

    paths = tuple(path for name in gated(record) for path in INPUTS[record][name])
    status, stdout, error = _git(root, "diff", "--name-only", tag, "--", *paths)
    if status != 0:
        return False, (
            f"`git diff {tag}` over the record's inputs could not be read "
            f"({error or f'git exited {status}'}), so the difference between version {named} and "
            f"this tree cannot be verified as the version header alone, and this check fails "
            f"closed")

    moved = {line.strip() for line in stdout.splitlines() if line.strip()}
    status, stdout, error = _git(root, "ls-files", "--others", "--exclude-standard", "--", *paths)
    if status != 0:
        return False, (
            f"the untracked files under the record's inputs could not be listed "
            f"({error or f'git exited {status}'}), so the difference between version {named} and "
            f"this tree cannot be verified as the version header alone, and this check fails "
            f"closed")
    moved.update(line.strip() for line in stdout.splitlines() if line.strip())

    beyond = sorted(moved - VERSION_METADATA)
    if beyond:
        return False, (
            f"The tree has moved from tag `{tag}` in {len(beyond)} file(s) the record reads beside "
            f"the version header: {', '.join(beyond[:6])}"
            + (", ..." if len(beyond) > 6 else "")
            + f", so the record's code has moved since the release it names")

    if VERSION_HEADER not in moved or not moved <= VERSION_METADATA:
        return False, (
            f"The tree reads the same content as tag `{tag}` over every file the record reads, so "
            f"the version header is not what moved; the record names version {named}, this tree "
            f"states another, and the two cannot be reconciled")

    status, header, error = _git_blob(root, f"{tag}:{VERSION_HEADER}")
    if status != 0:
        return False, (
            f"`{VERSION_HEADER}` at tag `{tag}` could not be read "
            f"({error or f'git exited {status}'}), so the record cannot be held to the release it "
            f"names, and this check fails closed")

    metadata = metadata_groups(record)
    for name in gated(record):
        expected = digest(
            root, INPUTS[record][name],
            {VERSION_HEADER: header} if name in metadata else None)
        stated = recorded.get(name, "")
        if not stated:
            return False, (
                f"The record's inputs hold no {name} group, so there is no content of that group "
                f"to compare against tag `{tag}` and the version difference cannot be verified")
        if stated != expected:
            return False, (
                f"The record's {name} content ({stated}) is not the content tag `{tag}` holds "
                f"({expected}), so the record is an account of a tree that release is not, "
                f"whatever its version line says")

    return True, ""


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
