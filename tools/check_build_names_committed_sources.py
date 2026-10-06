#!/usr/bin/env python3
"""Every source a build file names is a source the repository carries.

A CMake file may name a translation unit that exists in the working tree and was never
added to git. The configure step that reads it passes on the machine that wrote it and
**fails on every leg that clones the repository**, so the cost is one red CI run per
occurrence - and this has happened twice, both times in `CMakeLists.txt`, both times a
test registered by one worker while its sources sat untracked in another's tree.

The check is mechanical: read every CMake file's quoted source paths, and require each
one to be present **and tracked**. Untracked is the case that matters; absent is caught
by CMake itself, but it is caught here first and with a clearer sentence.

    python tools/check_build_names_committed_sources.py
    python tools/check_build_names_committed_sources.py --check    # exit 1 on a finding

Exit status: 0 when every named source is tracked; 1 when one is not, each named with
which of the two it is.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

SOURCE_SUFFIXES = (".cpp", ".cc", ".cxx", ".cu", ".hpp", ".h", ".cuh", ".inl")

# A source path inside a CMake file, quoted or bare. CMake names a target's sources
# bare (`add_executable(x tests/foo.cpp)`) and a definition's value quoted, so both are
# read. A line that is a comment is not: a path written in prose is not a file the build
# reads, and flagging one would make this check argue with documentation.
PATH_IN_LINE = re.compile(r'(?<![\w./-])([A-Za-z_][A-Za-z0-9_./-]*'
                          r'\.(?:cpp|cc|cxx|cu|hpp|h|cuh|inl))(?![\w-])')


def cmake_files(root: Path) -> list[Path]:
    def wanted(p: Path) -> bool:
        return p.name == "CMakeLists.txt" or p.suffix == ".cmake"

    return sorted(
        p for p in root.rglob("*")
        if wanted(p) and not any(part.startswith(("build", ".", "third_party")) for part in p.parts)
    )


def tracked(root: Path, relative: str) -> bool:
    found = subprocess.run(["git", "ls-files", "--error-unmatch", relative],
                           cwd=root, capture_output=True, text=True)
    return found.returncode == 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="exit 1 on any finding")
    options = parser.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    absent, untracked, named = [], [], 0

    for cmake in cmake_files(root):
        for line in cmake.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lstrip().startswith("#"):
                continue
            for found in PATH_IN_LINE.finditer(line):
                relative = found.group(1)
                named += 1
                if not (root / relative).is_file():
                    absent.append((cmake, relative))
                elif not tracked(root, relative):
                    untracked.append((cmake, relative))

    for cmake, relative in absent:
        print(f"{cmake.relative_to(root)}: names {relative}, which is not in the tree")
    for cmake, relative in untracked:
        print(f"{cmake.relative_to(root)}: names {relative}, which exists here but is NOT "
              f"tracked by git - every leg that clones this repository will fail to configure")

    total = len(absent) + len(untracked)
    print(f"check_build_names_committed_sources: {named} source path(s) named; "
          f"{total} not carried by the repository")

    if options.check and total:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
