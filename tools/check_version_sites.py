#!/usr/bin/env python3
"""Check that every prose site stating the library's version agrees with the project's number.

The version is written down once, in `CMakeLists.txt`'s `project(boys VERSION ...)` line, and
`include/boys/version.hpp` carries it to a program built without CMake. Both of those are held:
`tests/consumer_umbrella.cpp` fails when the header and the project disagree, and the two committed
runs are held by `check_recorded_run.py` and `check_probe_run.py`. The prose is not: `README.md`
states the version twice (the tree's number and the installed package file's compatibility), and
`docs/specification.md` and `CITATION.bib` each state it once. A release bump that moved the project
number and missed one of them would leave the front page or the citation describing another
release, and nothing would go red.

This check reads the version from `include/boys/version.hpp` through `tools/record_stamp.py` (the
same reader the recorded-run checks use; that header and `CMakeLists.txt`'s `project(...)` line are
held to each other by `tests/consumer_umbrella.cpp`) and holds these sites to it:

  * README.md             "This tree is **version X.Y.Z**."
  * README.md             "accepts any `X.x` package for a request of `X.Y.Z`" - both numbers, and
                          the package file is major-version compatible by construction
  * docs/specification.md "This tree is **version X.Y.Z**."
  * CITATION.bib          "version X.Y.Z, BSD-3-Clause"

A site whose sentence stops matching its pattern is a finding too: a checker that cannot find
what it was written to hold must not pass silently. The check reads prose and the citation note
only; the records' own version cells are the recorded-run checks' subject.

Usage:
  python3 tools/check_version_sites.py --check
  python3 tools/check_version_sites.py --root <dir>    # a negative control
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

TREE = re.compile(r"This tree is \*\*version (\d+\.\d+\.\d+)\*\*")
PACKAGE = re.compile(r"accepts any `(\d+)\.x` package for a request of `(\d+\.\d+\.\d+)`")
CITATION = re.compile(r"version (\d+\.\d+\.\d+), BSD-3-Clause")

SITES = (
    ("README.md", TREE, "the tree's version sentence"),
    ("README.md", PACKAGE, "the installed package file's compatibility sentence"),
    ("docs/specification.md", TREE, "the tree's version sentence"),
    ("CITATION.bib", CITATION, "the citation note's version"),
)


def record_stamp_module(root: pathlib.Path):
    """Load `tools/record_stamp.py` by path, so the version has one reader."""
    path = root / "tools" / "record_stamp.py"
    spec = importlib.util.spec_from_file_location("record_stamp", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(REPO), help="the repository to read")
    parser.add_argument("--check", action="store_true",
                        help="accepted for the invocation the workflow uses uniformly; this tool "
                             "has no other mode, so a disagreement exits non-zero with or without it")
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    findings: list[str] = []

    try:
        version = record_stamp_module(root).version(root)
    except Exception as error:  # noqa: BLE001 - the message is the finding
        print(f"STALE: the project's version could not be read from {root / 'CMakeLists.txt'}: "
              f"{error}")
        return 1

    print("the prose sites that state the library's version, held to version.hpp")
    print(f"  library version: {version}")

    version_match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version)
    if version_match is None:
        print(f"\nSTALE: the project's version {version!r} is not X.Y.Z, so no site can be held "
              f"to it")
        return 1
    major = int(version_match.group(1))

    for rel, pattern, what in SITES:
        path = root / rel
        text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        if not text:
            findings.append(f"{rel} is not a file this check can read, so {what} is held to "
                            f"nothing")
            continue

        matches = pattern.findall(text)
        if not matches:
            findings.append(f"{rel}: {what} was not found. A sentence this check holds removed or "
                            f"reworded leaves the version unguarded; restore the sentence or "
                            f"update this check with the site's new form")
            continue

        for match in matches:
            if isinstance(match, tuple):
                stated_major, stated = match
                agree = int(stated_major) == major and stated == version
                stated = f"{stated_major}.x / {stated}"
            else:
                stated = match
                agree = match == version

            if agree:
                print(f"  {rel}: {what} agrees")
            else:
                findings.append(
                    f"{rel}: {what} states {stated!r} and the project states {version}. A release "
                    f"that moved the project's number and left this site behind would ship a "
                    f"front page or a citation describing another release")

    if findings:
        print("\nSTALE: the prose version sites disagree with the project")
        for finding in findings:
            print(f"  - {finding}")
        return 1

    print(f"\nPASS: every prose version site agrees with the project's {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
