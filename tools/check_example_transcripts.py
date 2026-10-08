#!/usr/bin/env python3
"""Check the runs README.md and docs/getting-started.md quote against what the examples print.

`README.md` and `docs/getting-started.md` quote the output of `examples/NN_name.cpp` beside the code
that produced it, and `CMakeLists.txt` registers each example with ctest (`add_test(NAME
boys-example-${example} COMMAND boys-example-${example})`). That runs every program and compares
nothing, and no other tool reads an example's stdout: a change that moves a printed digit leaves
both documents quoting the old one and no run goes red. This is the reading step - it runs each
program and compares what it printed against the block the document quotes.

THE CONVENTION, EXACTLY - what the documents are relied on for

  * A DOCUMENT NAMES ITS PROGRAM by writing the file's path, `examples/NN_name.cpp`. That path is
    the only handle a quoted run and a program share. A name is looked for anywhere in the section,
    prose and block alike.

  * A QUOTED RUN is a code block the document presents as a program's output. Two shapes are
    recognized, and they are the two the documents use:
      - a fenced block with NO info string (README.md's shape), and
      - a block that follows a ```cpp fenced excerpt with nothing but blank lines between (the
        guide's shape: the excerpt shows the call, the block under it what it printed).
    A fenced block carrying a language (`cpp`, `cmake`) is source the document shows the reader and
    is never compared. Every other block - the shell commands both documents give, the guide's
    opening formula - is not a run, and each is listed in the report with that reason. Nothing is
    classified silently: every code block in every document appears in the output with its class.

  * A RUN BELONGS TO THE EXAMPLE ITS `##` SECTION NAMES. The section is the text from the block's
    own heading to the next heading. The two documents put the name in different places and both
    inside the section: the guide names the example in the section's opening line, above its
    excerpt and its run; README names it in the prose a few lines BELOW the run ("This program is
    [`examples/00_first_call.cpp`](examples/00_first_call.cpp)"). The name's position is not read -
    only which section it is in - because that is what makes the association decidable without
    parsing a human sentence. A section that names NO example, or names two different ones, is a
    FAILING finding and never a skip: the block is printed with its file and line and the run exits
    non-zero. That is the failure mode this check exists to avoid - a block nothing associates is a
    block nothing checks.

  * A DOCUMENT THAT STOPS QUOTING PROGRAMS FAILS, in two shapes, because both are how this kind of
    check goes quiet without anyone noticing: a document with no quoted run in it at all, and a
    section whose prose still names an example while its run is gone. A name inside a code block is
    not that claim - the guide's compile line names an example inside a shell command - so only
    prose namings are held to it.

  * WHITESPACE IS THE DOCUMENT'S, THE DIGITS ARE NOT. A document may wrap a quoted line or indent
    its block; the comparison is over the sequence of whitespace-separated tokens, so a re-wrapped
    block still matches and a document's alignment is not a figure. A token that differs is a
    failure and the two texts are printed side by side. Nothing is compared numerically:
    `0.055476132923077535` and `0.055476132923077536` are different strings and the check fails on
    them, which is the point - the documents quote 17 significant digits, so a last-bit change is a
    document that fails to reproduce.

  * ONE RUN IS NAMED RATHER THAN COMPARED, and named on every run. `examples/07_option_probe.cpp`
    times its own run against the host's clock. The document says of that block "Every figure in
    that transcript is a property of this host and this run", and it is right: the quoted leader,
    the size of the tied group and the nanosecond figures change between runs on one machine. The
    block is associated, counted, and printed as host-measured with that reason; it is not silently
    dropped, and every other run is compared.

USAGE

  python3 tools/check_example_transcripts.py --check --examples-dir build
  python3 tools/check_example_transcripts.py --check --build-dir build-win --config Release
  python3 tools/check_example_transcripts.py --check --doc /tmp/README.scratch.md --examples-dir build

Where the programs come from, in order of precedence: `--example NAME=PATH` for one program
spelled out; `--examples-dir DIR`, a directory of built programs searched as
`boys-example-NAME[.exe]` and in the `Release`, `RelWithDebInfo`, `Debug` and `MinSizeRel`
subdirectories a multi-configuration generator writes; `--build-dir DIR`, the same search with
`cmake --build DIR --target boys-example-NAME` run first for a program that is not there, so the
check does not require a tree someone built by hand. A program that is found nowhere is a failing
finding naming the example and the paths searched - never a skip. `--check` is the tree's spelling
for this tool's only mode; it is accepted so the call reads like its siblings.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DOCS = ("README.md", "docs/getting-started.md")

# The document's handle on its program, and the stem every other name is built from.
EXAMPLE_NAME = re.compile(r"examples/([0-9]{2}_[a-z0-9_]+)\.cpp")

# The configuration subdirectories a multi-configuration generator writes its executables into.
CONFIG_DIRS = ("Release", "RelWithDebInfo", "Debug", "MinSizeRel")

# Runs whose figures are the host's rather than the revision's, by example. Each entry is the reason,
# printed in the report on every run - a block named here is still associated, counted and shown, and
# only its token comparison is withheld.
HOST_MEASURED = {
    "07_option_probe": (
        "the program times its own run against the host's clock, and the document says of this "
        "block that every figure in it is a property of this host and this run: the leader, the "
        "length of the tied group and the nanosecond figures change between runs on one machine, so "
        "what the block quotes cannot be reproduced by any run, this tool's included"
    ),
}

HEADING = re.compile(r"^#{1,6}\s")
FENCE_OPEN = re.compile(r"^(\s*)```(.*)$")
FENCE_CLOSE = re.compile(r"^\s*```\s*$")
INDENTED = re.compile(r"^(?: {4,}|\t)\S")

# A program that hangs the leg is a failure with a name, not a leg that never finishes.
TIMEOUT_SECONDS = 900


@dataclass(frozen=True)
class Block:
    """One code block, 1-based and inclusive, with its content lines and their file line numbers."""

    start: int
    end: int
    kind: str  # "fenced" | "indented"
    tag: str  # a fenced block's info string; "" for an indented block
    lines: tuple[tuple[int, str], ...]

    @property
    def where(self) -> str:
        return f"line {self.start}" if self.start == self.end else f"lines {self.start}-{self.end}"


@dataclass(frozen=True)
class Token:
    """A whitespace-separated token of a run, with the line it was written on."""

    text: str
    line: int


@dataclass(frozen=True)
class Run:
    """A quoted run, the example its section names, and the program that answers for it."""

    doc: Path
    block: Block
    example: str


def scan_blocks(text: str) -> tuple[list[Block], list[str]]:
    """Every code block in a document, in file order, and anything that made the scan give up."""
    lines = text.split("\n")
    blocks: list[Block] = []
    problems: list[str] = []
    i = 0
    while i < len(lines):
        opening = FENCE_OPEN.match(lines[i])
        if opening:
            tag = opening.group(2).strip()
            j = i + 1
            while j < len(lines) and not FENCE_CLOSE.match(lines[j]):
                j += 1
            if j >= len(lines):
                problems.append(
                    f"line {i + 1}: a fenced block is opened and never closed, so everything "
                    f"below it is unread"
                )
                break
            content = tuple((k + 1, lines[k]) for k in range(i + 1, j))
            blocks.append(Block(i + 1, j + 1, "fenced", tag, content))
            i = j + 1
            continue

        if INDENTED.match(lines[i]):
            # An indented block is a block only where markdown makes one: after a blank line. An
            # indented line under prose is that paragraph's continuation, not the document's
            # presentation of a run.
            if i > 0 and lines[i - 1].strip():
                i += 1
                continue
            j = i
            last = i
            while j < len(lines) and (not lines[j].strip() or INDENTED.match(lines[j])):
                if lines[j].strip():
                    last = j
                j += 1
            content = tuple((k + 1, lines[k]) for k in range(i, last + 1))
            blocks.append(Block(i + 1, last + 1, "indented", "", content))
            i = j
            continue

        i += 1
    return blocks, problems


def classify(block: Block, previous: Block | None, lines: list[str]) -> tuple[str, str]:
    """What the document presents this block as, and the reason, for the report."""
    if block.kind == "fenced":
        if block.tag:
            return "source", f"a fenced block tagged `{block.tag}`: source the document shows"
        return "run", "a fenced block with no language: the document's shape for a quoted run"

    if previous is not None and previous.kind == "fenced" and previous.tag == "cpp":
        if all(not line.strip() for line in lines[previous.end:block.start - 1]):
            return "run", "it follows a `cpp` excerpt: the guide's shape for a quoted run"
    return "other", "an indented block that follows no `cpp` excerpt: the document's own text"


def section_bounds(headings: list[int], line: int, total: int) -> tuple[int, int]:
    """The section a line is in: its heading's line, and the line before the next heading."""
    start = 1
    for heading in headings:
        if heading <= line:
            start = heading
        else:
            return start, heading - 1
    return start, total


def tokenize(lines: tuple[tuple[int, str], ...]) -> list[Token]:
    tokens: list[Token] = []
    for number, text in lines:
        tokens.extend(Token(word, number) for word in text.split())
    return tokens


def tokenize_stdout(stdout: str) -> tuple[list[Token], dict[int, str]]:
    tokens: list[Token] = []
    by_line: dict[int, str] = {}
    for number, text in enumerate(stdout.split("\n"), start=1):
        by_line[number] = text.rstrip()
        tokens.extend(Token(word, number) for word in text.split())
    return tokens, by_line


def first_difference(quoted: list[Token], printed: list[Token]) -> int | None:
    """The index of the first token the two runs disagree on, or None when they agree entirely.

    The index can be one past the end of either run, which is how a document that stops early or
    carries on past the program is reported as a difference rather than as a prefix that matches.
    """
    common = min(len(quoted), len(printed))
    index = 0
    while index < common and quoted[index].text == printed[index].text:
        index += 1
    return None if index == len(quoted) == len(printed) else index


def locate(name: str, directories: list[Path], config: str | None) -> Path | None:
    """The built program for one example, in a directory or in a configuration subdirectory of it."""
    order = ([config] if config else []) + [c for c in CONFIG_DIRS if c != config]
    for directory in directories:
        for candidate in (directory / f"boys-example-{name}", directory / f"boys-example-{name}.exe"):
            if candidate.is_file():
                return candidate
        for sub in order:
            for candidate in (
                directory / sub / f"boys-example-{name}",
                directory / sub / f"boys-example-{name}.exe",
            ):
                if candidate.is_file():
                    return candidate
    return None


def build(build_dir: Path, name: str, config: str | None) -> str | None:
    """Build one example target in a build directory; the failure text, or None when it built."""
    command = ["cmake", "--build", str(build_dir), "--target", f"boys-example-{name}"]
    if config:
        command += ["--config", config]
    print(f"  not built yet: building `{' '.join(command)}`")
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError) as error:
        return f"`{' '.join(command)}` could not be run: {error}"
    if completed.returncode != 0:
        tail = "\n".join((completed.stderr or completed.stdout).strip().split("\n")[-8:])
        return f"`{' '.join(command)}` exited {completed.returncode}:\n{tail}"
    return None


def run_program(program: Path) -> tuple[str | None, str | None]:
    """Run one example program; its stdout, or the failure text."""
    try:
        completed = subprocess.run([str(program)], capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError) as error:
        return None, f"`{program}` could not be run: {error}"
    if completed.returncode != 0:
        return None, (
            f"`{program}` exited {completed.returncode}: the programs in examples/ decide for "
            f"themselves whether what they printed is sane, so this is one of them reporting a "
            f"defect rather than a transcript disagreeing with it"
        )
    return completed.stdout, None


def report_failure(
    doc: Path, doc_lines: list[str], run: Run, program: Path, quoted: list[Token],
    printed: list[Token], output_lines: dict[int, str], index: int,
) -> list[str]:
    """The lines this failure is reported with: the document's line and the program's, side by side."""
    header = (
        f"{doc}:{run.block.start} - the run of examples/{run.example}.cpp does not match what the "
        f"program printed"
    )
    if index < len(quoted) and index < len(printed):
        return [
            header,
            f"    document line {quoted[index].line}: {doc_lines[quoted[index].line - 1].strip()}",
            f"    program  line {printed[index].line}: {output_lines[printed[index].line]}",
            f"    token {index + 1}: the document quotes {quoted[index].text!r}, the program "
            f"printed {printed[index].text!r}",
            f"    program: {program}",
        ]
    if index >= len(printed):
        return [
            header,
            f"    the document quotes {len(quoted)} tokens and the program printed "
            f"{len(printed)}: the document carries on with {quoted[index].text!r} at its line "
            f"{quoted[index].line}",
            f"    document line {quoted[index].line}: {doc_lines[quoted[index].line - 1].strip()}",
            f"    program: {program}",
        ]
    return [
        header,
        f"    the document quotes {len(quoted)} tokens and the program printed {len(printed)}: "
        f"the program carries on with {printed[index].text!r} at line {printed[index].line} of its "
        f"output",
        f"    program  line {printed[index].line}: {output_lines[printed[index].line]}",
        f"    program: {program}",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true", help="verify (this tool's only mode)")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="the tree root: where the documents and examples/ live",
    )
    parser.add_argument(
        "--doc",
        type=Path,
        action="append",
        default=[],
        help=f"a document to read instead of the defaults ({', '.join(DEFAULT_DOCS)})",
    )
    parser.add_argument(
        "--example", action="append", default=[], metavar="NAME=PATH", help="one program, spelled out"
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        action="append",
        default=[],
        metavar="DIR",
        help="a directory of built programs, searched for boys-example-NAME[.exe]",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        action="append",
        default=[],
        metavar="DIR",
        help="a build directory; a program that is not there is built in it first",
    )
    parser.add_argument("--config", default=None, help="the configuration to look in and to build")
    args = parser.parse_args()

    root: Path = args.root
    docs = args.doc or [root / relative for relative in DEFAULT_DOCS]
    search_dirs: list[Path] = list(args.examples_dir) + list(args.build_dir)

    explicit: dict[str, Path] = {}
    for spelling in args.example:
        name, separator, path = spelling.partition("=")
        if not separator:
            print(f"check_example_transcripts: --example takes NAME=PATH, not {spelling!r}", file=sys.stderr)
            return 1
        explicit[name.strip()] = Path(path)

    missing_docs = [doc for doc in docs if not doc.is_file()]
    if missing_docs:
        print(
            "check_example_transcripts: no document at "
            + ", ".join(str(doc) for doc in missing_docs),
            file=sys.stderr,
        )
        return 1

    findings: list[str] = []
    runs: list[Run] = []
    contents: dict[Path, list[str]] = {}
    blocks_seen = 0
    sources_seen = 0
    others_seen = 0

    for doc in docs:
        lines = doc.read_text(encoding="utf-8", errors="replace").split("\n")
        contents[doc] = lines
        headings = [number for number, line in enumerate(lines, start=1) if HEADING.match(line)]
        blocks, problems = scan_blocks("\n".join(lines))
        findings.extend(f"{doc}:{problem}" for problem in problems)
        runs_here: list[Run] = []
        block_lines = {number for block in blocks for number, _ in block.lines}

        print(f"{doc}: {len(blocks)} code block(s)")
        for index, block in enumerate(blocks):
            blocks_seen += 1
            kind, reason = classify(block, blocks[index - 1] if index else None, lines)
            if kind == "source":
                sources_seen += 1
                print(f"  {block.where:<16} {reason}")
                continue
            if kind == "other":
                others_seen += 1
                print(f"  {block.where:<16} not a run: {reason}")
                continue

            start, end = section_bounds(headings, block.start, len(lines))
            named = sorted({m.group(1) for m in EXAMPLE_NAME.finditer("\n".join(lines[start - 1:end]))})
            if not named:
                findings.append(
                    f"{doc}:{block.start} - a quoted run in a section that names no example (the "
                    f"section starts at line {start}), so nothing associates it with a program. An "
                    f"unassociated block is a block nothing checks"
                )
                print(f"  {block.where:<16} run, UNASSOCIATED: the section names no example")
                continue
            if len(named) > 1:
                findings.append(
                    f"{doc}:{block.start} - a quoted run in a section that names {len(named)} "
                    f"examples ({', '.join(named)}), so no one program is its own"
                )
                print(f"  {block.where:<16} run, AMBIGUOUS: the section names {', '.join(named)}")
                continue

            name = named[0]
            if not (root / "examples" / f"{name}.cpp").is_file():
                findings.append(
                    f"{doc}:{block.start} - the section names examples/{name}.cpp, and there is no "
                    f"such file under {root / 'examples'}"
                )
            print(f"  {block.where:<16} run of examples/{name}.cpp, compared below")
            runs_here.append(Run(doc, block, name))

        # A section that names a program in its prose and quotes no run is the other half of an
        # unassociated block: the document still says a program answers this question and nothing
        # here reads what it prints. A name inside a code block is not that claim - the guide's
        # compile line names an example in a shell command - so only prose namings are held to it.
        for number, text in enumerate(lines, start=1):
            if number in block_lines:
                continue
            for named in sorted({match.group(1) for match in EXAMPLE_NAME.finditer(text)}):
                start, end = section_bounds(headings, number, len(lines))
                if any(run.block.start >= start and run.block.end <= end for run in runs_here):
                    continue
                findings.append(
                    f"{doc}:{number} - the prose here names examples/{named}.cpp and the "
                    f"section (from line {start}) quotes no run of it, so the document says a "
                    f"program answers this and nothing reads what it prints"
                )

        if not runs_here:
            findings.append(
                f"{doc} - not one quoted run was found in it: a document whose figures stop being "
                f"quoted there cannot fail this check, which is the way a check like this goes quiet"
            )
        runs.extend(runs_here)

    if not runs:
        print(
            "check_example_transcripts: not one quoted run was found in "
            + ", ".join(str(doc) for doc in contents)
            + ". A check that compares nothing is not a check",
            file=sys.stderr,
        )
        return 1

    # Resolve every program before running any of them, so a build that needs doing is done once.
    programs: dict[str, Path] = {}
    for name in sorted({run.example for run in runs}):
        program = explicit.get(name)
        if program is None:
            program = locate(name, search_dirs, args.config)
        if program is None:
            for build_dir in args.build_dir:
                failure = build(build_dir, name, args.config)
                if failure:
                    findings.append(f"examples/{name}.cpp - {failure}")
                    break
                program = locate(name, [build_dir], args.config)
                if program is not None:
                    break
        if program is None or not program.is_file():
            findings.append(
                f"examples/{name}.cpp - no built program for it: looked in "
                + (
                    ", ".join(str(directory) for directory in search_dirs)
                    if search_dirs
                    else "no directory at all, because --examples-dir and --build-dir are both empty"
                )
            )
            continue
        programs[name] = program

    outputs: dict[str, str] = {}
    for name, program in sorted(programs.items()):
        stdout, failure = run_program(program)
        if failure:
            findings.append(f"examples/{name}.cpp - {failure}")
        else:
            outputs[name] = stdout

    compared = 0
    host_measured = 0
    print()
    for run in runs:
        label = f"{run.doc} {run.block.where} run of examples/{run.example}.cpp"
        if run.example in HOST_MEASURED:
            host_measured += 1
            print(f"  {label}\n    HOST-MEASURED, not compared: {HOST_MEASURED[run.example]}")
            continue
        if run.example not in outputs:
            print(f"  {label}\n    NOT RUN: the program is reported below")
            continue

        program = programs[run.example]
        quoted = tokenize(run.block.lines)
        printed, output_lines = tokenize_stdout(outputs[run.example])
        index = first_difference(quoted, printed)
        if index is None:
            compared += 1
            print(f"  {label} - matches ({len(quoted)} tokens), {program}")
            continue

        findings.extend(report_failure(run.doc, contents[run.doc], run, program, quoted, printed, output_lines, index))
        print(f"  {label} - DOES NOT MATCH, reported below")

    print()
    print(
        f"check_example_transcripts: {len(contents)} document(s), {blocks_seen} code block(s): "
        f"{len(runs)} quoted run(s) of {len({run.example for run in runs})} example(s), "
        f"{compared} compared, {host_measured} host-measured, "
        f"{sources_seen} source block(s), {others_seen} not a run"
    )

    if findings:
        print("\ncheck_example_transcripts: the documents and the programs disagree:", file=sys.stderr)
        for finding in findings:
            print(f"\n  {finding}", file=sys.stderr)
        return 1

    print(
        f"check_example_transcripts: every one of the {compared} comparable quoted run(s) is what "
        f"its program printed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
