#!/usr/bin/env python3
"""A comment says what the code cannot, and then stops.

The rule this tool applies: comments are brief and exist only where they are needed; code that
explains itself is not commented, and documentation comments are the exception because
they are API rather than annotation. A comment that restates the line under it, or
that argues a case at length beside code that already makes it, is the defect.

What this checks is the part that can be counted: **a run of ordinary comment lines
longer than `--limit`**. A documentation comment is not counted — `///`, `/**`, `//!`
and a `/*!` block are API and are exempt by the rule itself — so what is left is a
non-documentation comment carrying more lines than an annotation needs.

    python tools/check_comments_are_brief.py             # the whole tree
    python tools/check_comments_are_brief.py --limit 4
    python tools/check_comments_are_brief.py --check     # exit 1 on any finding
    python tools/check_comments_are_brief.py path ...    # named files only

Exit status: 0 with no finding (or without `--check`); 1 when a run exceeds the limit
and `--check` was asked for.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SOURCE_SUFFIXES = (".cpp", ".cc", ".cxx", ".hpp", ".h", ".cu", ".cuh", ".inl")

# A line that is a comment at all, and the forms that are documentation rather than
# annotation. `///` and `//!` document the declaration below them; `/**` and `/*!`
# open a block that does. The rule exempts those, so they are not what is counted.
LINE_COMMENT = "//"
DOC_LINE = ("///", "//!")
BLOCK_OPEN = ("/*", "/**", "/*!")
DOC_BLOCK = ("/**", "/*!")


def comment_kind(line: str) -> str | None:
    """`doc`, `plain` or None for the line's own kind."""
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith("/*"):
        return "doc" if stripped.startswith(DOC_BLOCK) else "plain"
    if stripped.startswith(LINE_COMMENT):
        return "doc" if stripped.startswith(DOC_LINE) else "plain"
    return None


def runs(path: Path) -> list[tuple[int, int, bool]]:
    """Every run of comment lines: (first line, length, is_documentation)."""
    found: list[tuple[int, int, bool]] = []
    start, length, doc, in_block = 0, 0, False, False

    for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        kind = comment_kind(line)
        if kind is None and not in_block:
            if length:
                found.append((start, length, doc))
            start, length, doc = 0, 0, False
            continue
        if kind is None and in_block:
            length += 1  # the inside of a block comment
            if "*/" in line:
                in_block = False
            continue
        if not length:
            start, doc = number, (kind == "doc")
        length += 1
        if line.strip().startswith("/*") and "*/" not in line:
            in_block = True

    if length:
        found.append((start, length, doc))
    return found


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="files to read; default is the tree's sources")
    parser.add_argument("--limit", type=int, default=4,
                        help="the longest ordinary comment run allowed (default: %(default)s)")
    parser.add_argument("--check", action="store_true", help="exit 1 on any finding")
    options = parser.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    def wanted(p: Path) -> bool:
        if p.suffix not in SOURCE_SUFFIXES:
            return False
        # Every build tree is skipped by name, not by exact match: this repository
        # keeps one directory per writer and they are all called build-something.
        return not any(part.startswith(("build", ".", "third_party")) for part in p.parts)

    # A directory argument is expanded, not skipped. `--check include` used to read one path,
    # find nothing and exit 0 - a checker that reports success over a directory it never
    # opened, which two lanes hit before this line existed.
    named: list[Path] = []
    for argument in options.paths:
        path = Path(argument)
        named.extend(sorted(p for p in path.rglob("*") if wanted(p)) if path.is_dir() else [path])

    # A named path that selects nothing is said out loud, and is not a reason to read the whole
    # tree: `--check tools` names a directory of Python, none of it a source this reads, and the
    # fallback below answered it with the C++ tree's count - a number about files the caller did
    # not name, reported as if it were about the ones they did.
    if options.paths and not named:
        print(f"check_comments_are_brief: {', '.join(options.paths)} holds no source of the kinds "
              f"this reads ({' '.join(SOURCE_SUFFIXES)}), so no file was read", file=sys.stderr)
        return 1

    paths = named or sorted(p for p in root.rglob("*") if wanted(p))

    findings = []
    spliced = []

    for path in paths:
        if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
            continue
        for first, length, doc in runs(path):
            if not doc and length > options.limit:
                findings.append((path, first, length))

        # A `//` comment whose line ends in a backslash. The backslash splices the line into
        # the next one before the compiler reads it, so the comment swallows the line below -
        # and gcc says so as `-Wcomment: multi-line comment`, which the build's -Werror makes
        # fatal. Two seam fixtures carried one and no leg could compile them; this is the
        # sentence that would have said it before a CI run did.
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            stripped = line.rstrip()
            if stripped.lstrip().startswith(LINE_COMMENT) and stripped.endswith("\\"):
                spliced.append((path, number, stripped.strip()[:100]))

    def shown(path: Path) -> str:
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)

    for path, first, length in findings:
        print(f"{shown(path)}:{first}: {length} ordinary comment line(s)")

    for path, number, text in spliced:
        print(f"{shown(path)}:{number}: a line comment the backslash splices into the next line "
              f"- gcc refuses it under -Werror: {text}")

    total = len(findings) + len(spliced)
    print(f"check_comments_are_brief: {len(findings)} comment run(s) longer than "
          f"{options.limit} line(s) and {len(spliced)} spliced line comment(s) over "
          f"{len(paths)} file(s)")

    if options.check and total:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
