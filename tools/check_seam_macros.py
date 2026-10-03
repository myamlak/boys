#!/usr/bin/env python3
"""Check that every choice the build-defaults seam offers is one the library reads.

`include/boys/boys_build_defaults.hpp` defines the choices an entry that names no policy resolves
to, and `BOYS_BUILD_DEFAULTS` points a build at a replacement for it. The file is the contract a
replacement satisfies, and a choice written there and read by nothing is the defect this check
exists for: three of the five macros once sat in it, documented, while the library compiled
hard-coded literals instead. A consumer who replaced the file got two choices honoured and three
silently ignored, and nothing anywhere reported it.

The check reads the macro names off the seam file and requires each to be expanded somewhere in the
library's own sources - `include/` and `src/`, not counting the seam file itself, the test fixtures
that supply replacements for it, or the tooling. A macro with no reader is named and the run exits
non-zero.

What it does not check, and why: whether the reader is *correct*, and whether a reader that exists
is reached. Both are compile-time facts that a text check cannot see - a build that moves an axis
and fails to compile is the reading for those, and `tests/build_defaults_tuned.hpp` is where a moved
choice is exercised. This check is the cheap half: it fails when a choice stops being read at all,
which is the half that goes quiet.

Usage:
  python3 tools/check_seam_macros.py --check
  python3 tools/check_seam_macros.py --seam <file> --root <dir>    # a negative control
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

SEAM = pathlib.Path("include/boys/boys_build_defaults.hpp")

# Where a reader may live. The seam file itself is where the macros are defined and the fixtures are
# replacements for it, so neither counts as a reader.
READER_DIRS = ("include", "src")
NOT_READERS = (
    "include/boys/boys_build_defaults.hpp",
    "tests/build_defaults_noop.hpp",
    "tests/build_defaults_tuned.hpp",
)

DEFINE = re.compile(r"^\s*#\s*define\s+(BOYS_BUILD_DEFAULT_[A-Z_]+)\b", re.MULTILINE)
PREFIX = "BOYS_BUILD_DEFAULT_"


def readers(root: pathlib.Path) -> dict[str, list[str]]:
    """Every expansion of every seam macro, by macro name, in the library's own sources."""
    found: dict[str, list[str]] = {}

    for dirname in READER_DIRS:
        base = root / dirname
        if not base.is_dir():
            continue

        for path in sorted(base.rglob("*")):
            if path.suffix not in (".hpp", ".cpp", ".cu", ".h"):
                continue

            relative = path.relative_to(root).as_posix()
            if relative in NOT_READERS:
                continue

            text = path.read_text(encoding="utf-8", errors="replace")

            for match in re.finditer(rf"\b{PREFIX}[A-Z_]+\b", text):
                # The definition line is not a use.
                line = text[: match.start()].count("\n") + 1
                source_line = text.splitlines()[line - 1]
                if re.match(r"^\s*#\s*define\b", source_line):
                    continue
                found.setdefault(match.group(0), []).append(f"{relative}:{line}")

    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit non-zero on a seam defect")
    parser.add_argument("--seam", type=pathlib.Path, default=SEAM, help="the seam file to read")
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("."), help="the tree root")
    args = parser.parse_args()

    root = args.root
    seam = args.seam if args.seam.is_absolute() else root / args.seam

    if not seam.is_file():
        print(f"check_seam_macros: no seam file at {seam}", file=sys.stderr)
        return 1

    offered = DEFINE.findall(seam.read_text(encoding="utf-8"))
    if not offered:
        print(
            f"check_seam_macros: {seam} defines no {PREFIX}* macro. A seam that offers no choice is "
            f"not a seam, and a check that can pass by reading nothing is not a check",
            file=sys.stderr,
        )
        return 1

    used = readers(root)
    unread = [name for name in offered if name not in used]

    for name in offered:
        where = ", ".join(used.get(name, [])) or "nothing in include/ or src/ reads it"
        print(f"  {name:<38} {where}")

    if unread:
        print(f"\ncheck_seam_macros: {len(unread)} choice(s) offered and read by nothing:", file=sys.stderr)
        for name in unread:
            print(
                f"  {name}: defined in {seam} and expanded nowhere, so a build that replaced it "
                f"would be told the choice was taken and would compile something else",
                file=sys.stderr,
            )
        return 1

    print(f"\ncheck_seam_macros: every one of the {len(offered)} choice(s) the seam offers is read")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
