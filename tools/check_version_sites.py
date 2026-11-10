#!/usr/bin/env python3
"""Check that every prose site stating the library's version agrees with the project's number.

The version is written down once, in `CMakeLists.txt`'s `project(boys VERSION ...)` line, and
`include/boys/version.hpp` carries it to a program built without CMake. Both of those are held:
`tests/consumer_umbrella.cpp` fails when the header and the project disagree, and the two committed
runs are held by `check_recorded_run.py` and `check_probe_run.py`. The public prose sites do not
restate it: `README.md` and `docs/specification.md` name the declaration instead. The citation note
does name the release, because a citation places the software it cites.

This check reads the version from `include/boys/version.hpp` through `tools/record_stamp.py` (the
same reader the recorded-run checks use; that header and `CMakeLists.txt`'s `project(...)` line are
held to each other by `tests/consumer_umbrella.cpp`) and holds this site to it:

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

CITATION = re.compile(r"version (\d+\.\d+\.\d+), BSD-3-Clause")

SITES = (
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

    stamp = record_stamp_module(root)
    try:
        version = stamp.version(root)
    except Exception as error:  # noqa: BLE001 - the message is the finding
        print(f"STALE: the project's version could not be read from "
              f"{root / stamp.VERSION_HEADER}: {error}")
        return 1

    print("the prose sites that state the library's version, held to version.hpp")
    print(f"  library version: {version}")

    version_match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version)
    if version_match is None:
        print(f"\nSTALE: the project's version {version!r} is not X.Y.Z, so no site can be held "
              f"to it")
        return 1

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
            if match == version:
                print(f"  {rel}: {what} agrees")
            else:
                findings.append(
                    f"{rel}: {what} states {match!r} and the project states {version}. A release "
                    f"that moved the project's number and left this site behind would ship a "
                    f"citation describing another release")

    if findings:
        print("\nSTALE: the prose version sites disagree with the project")
        for finding in findings:
            print(f"  - {finding}")
        return 1

    print(f"\nPASS: every prose version site agrees with the project's {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
