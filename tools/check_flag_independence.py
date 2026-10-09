#!/usr/bin/env python3
"""The front door's values are the library's, not the calling unit's.

WHAT IT CHECKS. One source file, built several ways - a different optimization level, a different
target feature set, a different contraction setting each - must print the same digest of the bits
every default entry returned. The entries a call site names no policy for are defined once, in the
library, so nothing the caller's flags do can reach their arithmetic; a digest that differs says
they are being compiled at the call site instead, which is the defect.

An argument that is not exact in binary would make the program measure its own rounding rather
than the library's, so every argument is a multiple of 1/64 below 500 - the source says so where
it builds them.

    python tools/check_flag_independence.py --check build/boys-flag-independence-o2 ...
    python tools/check_flag_independence.py            # report only

Exit status: 0 when every program printed one digest, or when --check was not asked for; 1 when a
program failed to run or two digests differ, and 1 when fewer than two programs were named -
comparing one build against nothing is a vacuum and not a pass.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(program: Path) -> tuple[int, str]:
    """The program's digest, or its failure as an empty string."""
    completed = subprocess.run([str(program)], capture_output=True, text=True)
    digest = completed.stdout.strip()
    if completed.returncode != 0 or not digest:
        print(f"{program.name}: exited {completed.returncode}, printed {digest!r}")
        if completed.stderr.strip():
            print(completed.stderr.strip()[:400])
        return completed.returncode or 1, ""
    return 0, digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero when the digests are not all the same")
    parser.add_argument("programs", nargs="+", type=Path,
                        help="the same translation unit built under different flags")
    arguments = parser.parse_args()

    if len(arguments.programs) < 2:
        print("check_flag_independence: fewer than two builds were named, so there is nothing "
              "to compare and nothing this run could have found")
        return 1

    digests: list[tuple[str, str]] = []
    for program in arguments.programs:
        status, digest = run(program)
        if status != 0:
            return 1
        digests.append((program.name, digest))
        print(f"{program.name}: {digest}")

    distinct = sorted({digest for _, digest in digests})
    if len(distinct) == 1:
        print(f"PASS: {len(digests)} builds under different flags printed one digest, "
              f"{distinct[0]}")
        return 0

    print(f"FAIL: {len(digests)} builds printed {len(distinct)} different digests: "
          f"{', '.join(distinct)}")
    for name, digest in digests:
        print(f"  {name}: {digest}")
    return 1 if arguments.check else 0


if __name__ == "__main__":
    sys.exit(main())
