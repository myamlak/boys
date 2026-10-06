#!/usr/bin/env python3
"""Compile the sources a change touches, with the compilers CI uses.

MSVC is greener than CI, and the difference has cost this project red legs five
times: an internal function defined and never emitted (clang), a variable set and
never read (gcc), an expression nested past clang's bracket depth. Every one was a
source that compiled on the machine it was written on and failed on the legs that
decide the merge. A build here is minutes; this is seconds per file, and it reads
only the files named, so it fits between an edit and a commit.

The flags are not guessed. Each source is compiled with the command the build itself
uses for that source - the entry `compile_commands.json` holds - with the compiler
swapped for the one asked for and the output dropped for `-fsyntax-only`. So a file
is checked at the flags, defines and include path it ships at, and a warning this
build treats as an error is an error here too.

Usage:

    python tools/check_compiles_here.py                    # the tree's own diff
    python tools/check_compiles_here.py tests/boys_x.cpp   # named files
    python tools/check_compiles_here.py --compiler clang   # one compiler only

Exit status: 0 when every file compiles under every compiler asked for; 1 when at
least one does not, each named with the compiler's own first diagnostic; 2 when a
file named has no compile command at all - a source no target builds is reported
rather than skipped, because skipping it is how a file comes to be committed that
nothing compiles.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

# The compilers CI builds with. A source is checked under each.
COMPILERS = {
    "gcc": "g++",
    "clang": "clang++",
}

# What a source file we compile looks like. A file of another kind is not ours to
# check directly.
SOURCE_SUFFIXES = (".cpp", ".cc", ".cxx", ".cu")

# What is compiled into something else rather than on its own. A header in a change
# is checked through the translation units that name it.
HEADER_SUFFIXES = (".hpp", ".h", ".cuh", ".inl")

# The first line of a compiler's diagnostic, so a failure reads as one line rather
# than as a screenful. Whatever follows the first `error:`/`warning:` is kept.
DIAGNOSTIC = re.compile(r"^(.*?:(?:error|fatal error|warning):.*)$", re.MULTILINE)

# The limits CI's compilers impose that a newer local one may have raised. A check
# that inherits this machine's defaults answers a question about this machine, and
# the question is about the legs: clang 21's bracket depth is deeper than clang 18's,
# so a 1152-argument fold that fails CI compiled clean here until this was stated.
CI_LIMITS = {
    "clang++": ["-fbracket-depth=256"],
}

# How many translation units a changed header is checked through. A header included
# everywhere is compiled many times over; the first few that name it settle whether
# it compiles at all, and the whole tree is a build's job rather than this one's.
HEADER_TU_LIMIT = 3

# The build directories read when none is named. The second is the CUDA one, and it is
# the second for a reason: a source built only there is a device source, and reading
# the non-CUDA directory alone says "no target builds it" about every one of them.
DEFAULT_BUILD_DIRS = ("build-checkgate", "build-checkgate-cuda")


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile the named sources with the compilers CI uses.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="sources to check; default is the working tree's own change against HEAD",
    )
    parser.add_argument(
        "--build-dir",
        action="append",
        help="a configured build directory carrying compile_commands.json; repeatable, "
        "and one configured WITH CUDA is what checks the device sources. "
        "(default: build-checkgate, then build-checkgate-cuda if it exists)",
    )
    parser.add_argument(
        "--compiler",
        action="append",
        choices=sorted(COMPILERS),
        help="which compiler to check with; repeatable (default: all of them)",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="the repository root (default: the directory holding this script's parent)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="how many compiles to run at once (default: one per processor, at most 4)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print every command run, not only the ones that fail",
    )
    return parser.parse_args(argv)


def repo_root(named: str | None) -> Path:
    if named is not None:
        return Path(named).resolve()
    return Path(__file__).resolve().parent.parent


def load_commands(build_dir: Path) -> dict[Path, list[dict]]:
    """Every source's compile command, keyed by the source's resolved absolute path.

    A file may be built into more than one target and so carry more than one command;
    all of them are kept, because the file must compile in each.
    """
    manifest = build_dir / "compile_commands.json"
    if not manifest.is_file():
        raise SystemExit(
            f"{manifest} does not exist: configure the build with "
            f"-DCMAKE_EXPORT_COMPILE_COMMANDS=ON and name it with --build-dir"
        )

    entries = json.loads(manifest.read_text(encoding="utf-8"))
    keyed: dict[Path, list[dict]] = {}

    for entry in entries:
        directory = Path(entry["directory"])
        source = Path(entry["file"])
        if not source.is_absolute():
            source = directory / source
        keyed.setdefault(source.resolve(), []).append(entry)

    return keyed


def keyed_for_header(
    header: Path, keyed: dict[Path, list[dict]], root: Path
) -> list[dict]:
    """The compile commands of the translation units whose source names this header.

    The name is read as the sources spell it - `#include <boys/boys_x.hpp>` or a
    relative path - so both spellings are looked for, and the search is over the
    sources the build knows about rather than over the tree.
    """
    names = {header.name, f"{header.parent.name}/{header.name}"}
    for candidate in (header, *header.parents):
        if candidate == root or root not in candidate.parents:
            break
        names.add(str(header.relative_to(candidate)).replace("\\", "/"))

    found: list[dict] = []
    for source, entries in keyed.items():
        if not source.is_file() or source.suffix not in SOURCE_SUFFIXES:
            continue
        try:
            text = source.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not any(f'"{name}"' in text or f"<{name}>" in text for name in names):
            continue
        found.extend(entries)
        if len(found) >= HEADER_TU_LIMIT:
            break

    return found[:HEADER_TU_LIMIT]


def command_of(entry: dict) -> list[str]:
    if "arguments" in entry:
        return list(entry["arguments"])
    return shlex.split(entry["command"])


def is_cuda(command: list[str]) -> bool:
    return "nvcc" in Path(command[0]).name


def check_command(command: list[str], compiler: str | None) -> list[str]:
    """The build's own command, retargeted at `compiler` and made a syntax check."""
    cuda = is_cuda(command)
    checked: list[str] = []
    index = 0
    while index < len(command):
        word = command[index]
        if index == 0 and compiler is not None:
            checked.append(compiler)
        elif word == "-o":
            index += 1  # drop the output path and its argument with it
        elif word.startswith("-o") and len(word) > 2:
            pass
        elif word in ("-c", "--compile"):
            # nvcc has no -fsyntax-only - it answers `Unknown option` and exits 1 - so a
            # CUDA source is checked by compiling it. The -c stays and the object is sent
            # to the null device below. Dropping -c makes nvcc link, and a translation unit
            # with no main dies on `undefined reference to 'main'`: a failure of this
            # command line, read as a failure of the source, which is why no .cu change
            # could be committed while this was the behaviour.
            if cuda:
                checked.append(word)
        else:
            checked.append(word)
        index += 1

    if cuda:
        checked.extend(["-o", os.devnull])

    if compiler is not None:
        checked.extend(CI_LIMITS.get(Path(compiler).name, ()))

    if "-fsyntax-only" not in checked and not cuda:
        checked.append("-fsyntax-only")
    return checked


def first_diagnostic(output: str) -> str:
    found = DIAGNOSTIC.search(output)
    if found is not None:
        return found.group(1).strip()
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[0] if lines else "(no diagnostic)"


def run_one(job: tuple[Path, list[str], str | None, bool, Path]) -> tuple[Path, str | None, str]:
    """Compile one source once. Returns (source, compiler label, diagnostic-or-empty)."""
    source, command, compiler, verbose, directory = job
    label = Path(compiler).name if compiler else "the build's own compiler"

    if verbose:
        print(f"  {label} {' '.join(command)}", file=sys.stderr)

    finished = subprocess.run(
        command,
        cwd=directory,
        capture_output=True,
        text=True,
        errors="replace",
    )

    if finished.returncode == 0:
        return source, compiler, ""

    return source, compiler, first_diagnostic(finished.stdout + finished.stderr)


def main(argv: list[str]) -> int:
    options = parse_arguments(argv)
    root = repo_root(options.repo)

    # More than one build directory is the normal case: the device sources are built
    # only by a build configured with CUDA, and a check that reads the non-CUDA one
    # alone reports every device source as unbuilt - which is how a gate comes to have
    # a blind spot exactly where the defects are.
    build_dirs = [Path(name) for name in (options.build_dir or DEFAULT_BUILD_DIRS)]
    build_dirs = [name if name.is_absolute() else root / name for name in build_dirs]

    wanted = [Path(name).resolve() for name in options.files]
    if not wanted:
        wanted = default_files(root)

    wanted = [
        name for name in wanted if name.suffix in SOURCE_SUFFIXES + HEADER_SUFFIXES
    ]
    if not wanted:
        print("check_compiles_here: no source to check", file=sys.stderr)
        return 0

    keyed: dict[Path, list[dict]] = {}
    read_dirs: list[Path] = []
    for directory in build_dirs:
        if not (directory / "compile_commands.json").is_file():
            continue
        for source, entries in load_commands(directory).items():
            keyed.setdefault(source, []).extend(entries)
        read_dirs.append(directory)

    if not read_dirs:
        raise SystemExit(
            f"no build directory among {', '.join(str(d) for d in build_dirs)} carries "
            f"compile_commands.json: configure one with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
        )

    chosen = options.compiler or sorted(COMPILERS)

    jobs: list[tuple[Path, list[str], str | None, bool, Path]] = []
    unbuilt: list[Path] = []

    for source in wanted:
        entries = keyed.get(source)
        if not entries and source.suffix in HEADER_SUFFIXES:
            # A header is not compiled on its own: it is compiled into every translation
            # unit that names it. Checking those is what checks the header, and it is
            # also the closest thing to what the build will do to it.
            entries = keyed_for_header(source, keyed, root)
            if entries:
                print(
                    f"check_compiles_here: {source.relative_to(root)} has no target of its "
                    f"own; checked through {len(entries)} translation unit(s) that include it",
                    file=sys.stderr,
                )
        if not entries:
            unbuilt.append(source)
            continue
        for entry in entries:
            command = command_of(entry)
            directory = Path(entry["directory"])
            for name in chosen:
                # A CUDA source keeps nvcc: the host compilers do not read it, and
                # the legs that failed on it were nvcc's own.
                compiler = None if is_cuda(command) else COMPILERS[name]
                checked = check_command(command, compiler)
                jobs.append((source, checked, compiler, options.verbose, directory))

    if unbuilt:
        for source in unbuilt:
            print(
                f"check_compiles_here: {source} has no compile command in "
                f"{', '.join(d.name for d in read_dirs)}/compile_commands.json - no "
                f"target among them builds it",
                file=sys.stderr,
            )
        return 2

    workers = options.jobs if options.jobs > 0 else min(4, os.cpu_count() or 1)
    failures: list[tuple[Path, str, str]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for source, compiler, diagnostic in pool.map(run_one, jobs):
            if diagnostic:
                label = Path(compiler).name if compiler else "nvcc"
                failures.append((source, label, diagnostic))

    for source, label, diagnostic in failures:
        print(f"FAIL {source.relative_to(root)} [{label}]: {diagnostic}", file=sys.stderr)

    checked = len(jobs)
    if failures:
        print(
            f"check_compiles_here: {len(failures)} of {checked} compile(s) failed",
            file=sys.stderr,
        )
        return 1

    print(f"check_compiles_here: {checked} compile(s) clean over {len(wanted)} source(s)")
    return 0


def default_files(root: Path) -> list[Path]:
    """What the working tree changes against HEAD, staged or not."""
    changed = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        cwd=root,
        capture_output=True,
        text=True,
    )

    names = set(changed.stdout.split()) | set(staged.stdout.split())
    return [(root / name).resolve() for name in sorted(names)]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
