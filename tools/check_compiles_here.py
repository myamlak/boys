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
# check - and a header is checked through the sources that include it.
SOURCE_SUFFIXES = (".cpp", ".cc", ".cxx", ".cu")

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
        default="build-checkgate",
        help="a configured build directory carrying compile_commands.json "
        "(default: %(default)s)",
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


def command_of(entry: dict) -> list[str]:
    if "arguments" in entry:
        return list(entry["arguments"])
    return shlex.split(entry["command"])


def is_cuda(command: list[str]) -> bool:
    return "nvcc" in Path(command[0]).name


def check_command(command: list[str], compiler: str | None) -> list[str]:
    """The build's own command, retargeted at `compiler` and made a syntax check."""
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
            pass
        else:
            checked.append(word)
        index += 1

    if compiler is not None:
        checked.extend(CI_LIMITS.get(Path(compiler).name, ()))

    if "-fsyntax-only" not in checked and not is_cuda(command):
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
    build_dir = Path(options.build_dir)
    if not build_dir.is_absolute():
        build_dir = root / build_dir

    wanted = [Path(name).resolve() for name in options.files]
    if not wanted:
        wanted = default_files(root)

    wanted = [name for name in wanted if name.suffix in SOURCE_SUFFIXES]
    if not wanted:
        print("check_compiles_here: no source to check", file=sys.stderr)
        return 0

    keyed = load_commands(build_dir)
    chosen = options.compiler or sorted(COMPILERS)

    jobs: list[tuple[Path, list[str], str | None, bool, Path]] = []
    unbuilt: list[Path] = []

    for source in wanted:
        entries = keyed.get(source)
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
                f"{build_dir.name}/compile_commands.json - no target builds it",
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
