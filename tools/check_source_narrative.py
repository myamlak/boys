#!/usr/bin/env python3
"""Shipped text must not narrate the project to a reader who was not there.

WHAT IT CHECKS. Every line of the shipped text under `include/`, `src/`, `docs/`, `tools/` and
`tests/` against the phrase table below: what a thing was before this change, what is still owed,
what will replace it; the author's own first person; and a path into a machine only a maintainer
has. An audit of this tree found all three committed. A reader of a public header was not there for
the first, is not the "I" of the second, and cannot open the path of the third.

    python tools/check_source_narrative.py               # the tree, report only
    python tools/check_source_narrative.py --check       # exit non-zero on any finding
    python tools/check_source_narrative.py --control     # plant the defect; require it caught
    python tools/check_source_narrative.py path ...      # named files or directories

Exit status: 0 when the text is clean, and 0 when findings are printed without `--check`; 1 when a
finding is printed and `--check` was asked for; 1 when `--control` fails to catch its own fixture,
or when the run reads no file at all.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SELF = Path(__file__).resolve()

# The shipped text this reads. A directory is skipped by prefix or by name: `build*` is a build tree
# in every checkout, a dot-directory is not shipped, and the vendored tree is not ours.
ROOTS = ("include", "src", "docs", "tools", "tests")
SKIPPED_BY_PREFIX = (".", "build")
SKIPPED_BY_NAME = ("third_party", "__pycache__")

# The phrases the shipped text may not carry, one group per reason it is a defect. An entry is a
# regular expression matched case-insensitively against the whole line, so a phrase whose case
# matters says so itself. A group is extended by adding a line to it.
PHRASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Process and history. A public reader did not watch the project get here: "no longer", "this
    # revision", "is owed" and "stops moving" each describe the state of the work rather than a fact
    # about the library, and each dates the sentence to a revision the reader cannot open.
    (
        "process/history",
        (
            r"this revision",
            r"until this revision",
            r"no longer",
            r"previously",
            r"used to be",
            r"was the second",
            r"is owed",
            r"will be replaced",
            r"stops moving",
            r"so far",
        ),
    ),
    # First person and development narrative. "I", "my" and "we" address the reader as the author;
    # "the owner" and "the maintainer" name people a reader has no way to know; "for now" and "at
    # present" date a sentence the same way the group above does. The pronoun is case-sensitive and
    # no character of a code token may touch it, so the `-I` flag, the `I/O` lane, the `re.I` flag
    # and the `"<I"` struct format are read as the code they are and not as the author.
    (
        "first person",
        (
            r"""(?-i:(?<![-\w/.<>=])I(?![/\w]))""",
            r"\bmy\b",
            r"\bwe\b",
            r"\bthe owner\b",
            r"\bthe maintainer\b",
            r"\bfor now\b",
            r"\bat present\b",
        ),
    ),
    # A pointer into the machine the text was written on. `.claude/` and a lane's status file live
    # under a maintainer's working directory: a reader of a shipped header cannot open one, so text
    # that points at one is text about the project and not about the library.
    (
        "machine pointer",
        (
            r"\.claude/",
            r"\.claude/tmp",
            r"lane-status",
        ),
    ),
)

COMPILED = tuple(
    (label, tuple((source, re.compile(source, re.IGNORECASE)) for source in patterns))
    for label, patterns in PHRASES
)

# The control, planted in memory rather than read from a file. Every line but the last carries the
# defect of one group; the last is ordinary prose that must not be named. A table that matches every
# line catches this fixture too, and a table that matches nothing fails it - which is the vacuum.
CONTROL = (
    '/// the fit granularity was the second until this revision, and the figure is owed\n'
    '/// an operator meant "I " by it, the reading is my own, and we keep it for now\n'
    '/// read from .claude/tmp/lane-status/probes/host-defaults-emitted.hpp, so far\n'
    "/// The region is kDefaultRegionBExp, and the member it names is the one returned.\n"
)


def hits(text: str) -> list[tuple[int, str, str, str]]:
    """Every (line number, group, phrase, line) of `text` the table matches, in line order."""
    return [
        (number, label, source, line.rstrip())
        for number, line in enumerate(text.splitlines(), 1)
        for label, patterns in COMPILED
        for source, pattern in patterns
        if pattern.search(line)
    ]


# What the rule is about: an author's voice talking about the project's own process, which a reader
# meets in a comment, a docstring or a document. It is NOT about a string the program prints - the
# gate's own verdict line reads "every documented claim met at this revision", and that sentence is
# the gate's output and not its author's aside. Policing printed strings would make this checker
# argue with a run, so a string literal is blanked before the table is applied.
CODE_SUFFIXES = (".cpp", ".cc", ".cxx", ".hpp", ".h", ".cu", ".cuh", ".inl", ".py")
DOCSTRING_SUFFIX = ".py"


def prose_only(text: str, suffix: str) -> str:
    """The file's prose, with every other character blanked so that line numbers do not move.

    A document is prose in its own right and is returned as it stands. A source file keeps its
    comments and its Python docstrings, and loses its string literals - every character of one
    becoming a space, so a finding still points at the line the reader would look at.
    """
    if suffix not in CODE_SUFFIXES:
        return text

    out = list(text)
    size = len(text)

    def blank(start: int, stop: int) -> None:
        for at in range(start, min(stop, size)):
            if out[at] != "\n":
                out[at] = " "

    at = 0
    while at < size:
        if text.startswith("//", at):
            end = text.find("\n", at)
            end = size if end < 0 else end
            blank(at, end)
            at = end
        elif text.startswith("/*", at):
            end = text.find("*/", at + 2)
            end = size if end < 0 else end + 2
            blank(at, end)
            at = end
        elif suffix == DOCSTRING_SUFFIX and text.startswith(('"""', "'''"), at):
            # Kept, not blanked: a Python triple-quoted string is overwhelmingly a docstring, and a
            # docstring is prose this rule is about. The exception is a literal that is not one, and
            # that is a finding a reader can see rather than a silent miss.
            quote = text[at:at + 3]
            end = text.find(quote, at + 3)
            at = size if end < 0 else end + 3
        elif suffix == DOCSTRING_SUFFIX and text[at] == "#":
            end = text.find("\n", at)
            end = size if end < 0 else end
            blank(at, end)
            at = end
        elif text[at] == '"' or (text[at] == "'" and suffix != DOCSTRING_SUFFIX):
            quote = text[at]
            cursor = at + 1
            while cursor < size and text[cursor] != quote:
                cursor += 2 if text[cursor] == "\\" else 1
            blank(at, min(cursor + 1, size))
            at = min(cursor + 1, size)
        else:
            at += 1

    return "".join(out)


def text_of(path: Path) -> str | None:
    """The file's text, or None for a file that is not text at all."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return None if b"\0" in data else data.decode("utf-8", errors="replace")


def unskipped(path: Path) -> bool:
    """Whether the file is read at all: not a build tree, and not this tool's own phrase table.

    The table is the one text in the tree that carries every phrase it looks for, so it is skipped
    by name; the run says so rather than letting the omission read as a clean file.
    """
    if path.resolve() == SELF:
        return False
    return not any(
        part.startswith(SKIPPED_BY_PREFIX) or part in SKIPPED_BY_NAME for part in path.parts[:-1]
    )


def shipped(path: Path) -> bool:
    """Whether the default walk reads the file: under a root of this tree, and not otherwise skipped.

    `tests/data/` is not read. It holds what a run wrote - the gate's recorded run, the probe
    reports - and a record is re-made rather than edited, so a phrase rule applied there would ask
    for an edit that the next run undoes.
    """
    try:
        parts = path.relative_to(REPO).parts
    except ValueError:
        return False

    if parts[:2] == ("tests", "data"):
        return False

    return parts[0] in ROOTS and unskipped(path)


def tree_paths() -> tuple[list[Path], str]:
    """The shipped text of the tree, and the scope it was taken with.

    Tracked files are the shipped text: another writer's untracked scratch is not in the tree, and a
    count over a working tree that has some is a count nobody else can reproduce. A copy of the tree
    without git history is read whole instead, and the run says which scope it took.
    """
    try:
        done = subprocess.run(["git", "-C", str(REPO), "ls-files"], capture_output=True,
                              text=True, timeout=60)
    except OSError:
        done = None

    if done is not None and done.returncode == 0:
        return sorted(p for p in (REPO / name for name in done.stdout.splitlines())
                      if shipped(p) and p.is_file()), "tracked"

    return sorted(p for root in ROOTS for p in (REPO / root).rglob("*")
                  if shipped(p) and p.is_file()), "on-disk"


def control() -> int:
    """Read the planted fixture and require every planted line named and the clean line spared."""
    lines = CONTROL.splitlines()
    planted = set(range(1, len(lines)))
    found = hits(CONTROL)
    named = {number for number, _, _, _ in found}

    for number, _label, phrase, line in found:
        print(f"control:{number}: {phrase} — {line}")

    problems = []
    if planted - named:
        problems.append(f"line(s) {sorted(planted - named)} carry a phrase and were not named")
    if len(lines) in named:
        problems.append(f"the clean line {len(lines)} was named")
    if problems:
        print("check_source_narrative: the control failed: " + "; ".join(problems), file=sys.stderr)
        return 1

    print(f"check_source_narrative: control caught all {len(planted)} planted line(s) and left "
          f"the clean line alone")
    return 0


def main(argv: list[str]) -> int:
    for stream in (sys.stdout, sys.stderr):
        # A line of shipped text can carry a character this console's code page has no glyph for,
        # and a checker that dies printing its finding reports nothing.
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*",
                        help="files or directories to read; default is the tree's own text")
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero when the text carries a phrase from the table")
    parser.add_argument("--control", action="store_true",
                        help="read the planted fixture instead of the tree, and exit non-zero "
                             "unless every planted line is caught and the clean one is not")
    options = parser.parse_args(argv)

    if options.control:
        return control()

    named = [Path(argument) for argument in options.paths]
    if named:
        paths = sorted(p for argument in named
                       for p in (argument.rglob("*") if argument.is_dir() else [argument])
                       if p.is_file() and unskipped(p))
        scope = "named"
    else:
        paths, scope = tree_paths()

    findings = []
    read = 0
    for path in paths:
        text = text_of(path)
        if text is None:
            continue
        read += 1
        findings.extend((path, *finding) for finding in hits(prose_only(text, path.suffix)))

    if not read:
        print("check_source_narrative: no file was read - a run over nothing is not a clean tree",
              file=sys.stderr)
        return 1

    for path, number, _label, phrase, line in findings:
        try:
            shown = path.relative_to(REPO)
        except ValueError:
            shown = path
        print(f"{shown}:{number}: {phrase} — {line}")

    lines = len({(path, number) for path, number, _, _, _ in findings})
    files = len({path for path, *_ in findings})
    counts = {label: sum(1 for _, _, group, _, _ in findings if group == label)
              for label, _ in PHRASES}
    print(f"check_source_narrative: {len(findings)} phrase hit(s) over {lines} line(s) in {files} "
          f"of {read} {scope} text file(s) read; this tool's own file is never read, it carries "
          f"the table")

    for label, _ in PHRASES:
        print(f"  {label:<16} {counts[label]} hit(s)")

    return 1 if options.check and findings else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
