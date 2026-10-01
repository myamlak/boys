#!/usr/bin/env python3
"""Check the runs the documents quote against the source strings they quote.

`LaneContractInfo`'s `source` member is the library's own words for the terms a
lane's base figure does not carry, and every row's value is text a consumer
document quotes, because a program prints it: `examples/01_one_order.cpp` prints
the field as `(from %s)`, `examples/06_what_it_guarantees.cpp` prints it as
`[%s]` in a table column, and docs/lane-contract.md restates it in the
`bound: ...` cell of a row that names the lane.

A field that grows while a document keeps the text it used to carry leaves the
document describing output the programs no longer produce. That is not
hypothetical: the fp64 and fp32 rows were edited into 583- and 1123-character
rationale paragraphs while `docs/getting-started.md` went on quoting the
24-character sentence those rows had said, so the examples printed walls into a
table column and the documents were wrong about what they print. Nothing in the
tree compared a quoted run against the string it quotes.

This script does. It reads the rows' `source` strings out of src/boys.cpp and
every run of those three shapes out of README.md, docs/getting-started.md and
docs/lane-contract.md, and compares them after collapsing each side's
whitespace - a document may wrap a run across lines and the field has none, and
a line break is a document's, not a quotation's. It then reports three kinds of
run:

  * a run that is a proper prefix of a source string and at least twelve
    characters long is a MISMATCH, and it fails the run. That is the shape the
    drift took: the document carries the text it once quoted and the field
    writes on past it. The finding names the row whose string the run begins,
    how many of that string's characters it carries, and what the field says
    next, so the reader has the two texts rather than a verdict about them.

  * a run equal to a source string is a confirmed quote, named and counted.

  * every other run is one this check says nothing about, counted by shape and
    not listed. The documents use the same brackets for cross-reference labels,
    the same parentheses for shell text and the same table cell for their own
    prose; holding those to a field of the library would be a claim the check
    cannot support, and it makes none.

Why the rule faces one way. A run the field begins with and writes past is
decidable from the two texts: one stopped, the other did not. A run *longer*
than every source string is not. A paraphrase ("the lane's documented figure,
plus ...") and a quotation of a field that has since been shortened read the
same to any comparison of the two texts, and this script does not guess between
them; where a run begins with a whole source string and continues, that is
counted and printed and never failed.

Why twelve characters. The shortest string the rows carry is a full clause, so
a run agreeing with a source string character for character is a fragment of a
sentence rather than a word. Twelve is the briefest run worth reading as a
quotation; a shorter agreement is counted with the unconfirmed runs and not
failed.

The unit of comparison is the field, not a captured transcript. The two shapes
a program prints carry the field through `%s`, so there the field's characters
and the program's output are the same characters. The `bound: ...` cells are
compared against the field as well, and they restate it beside a figure rather
than quoting a print, which is why a run there that is no part of the field is
read as the table's own prose.

What this check does not read, and why:

  * the `delivered: ...` half of those cells, whose runs carry markdown
    backticks the field does not have, and the delivery accessor's `source`
    strings behind them, which name the tables a figure was read from rather
    than terms the base does not carry and reach no document in one of the
    three shapes above;

  * `examples/`, whose output is this check's subject rather than an input;

  * the figures the same lines print. `tools/check_doc_arithmetic.py` reads the
    arithmetic of the prose and the accuracy gate reads the bounds; a figure
    beside a run is neither this check's subject nor one it can confirm.

Usage:
    python tools/check_doc_transcripts.py --check
    python tools/check_doc_transcripts.py --check --source /tmp/old_boys.cpp

`--source` reads the strings from another file, which is how the pre-restoration
tree is reproduced as a negative control: `git show <rev>:src/boys.cpp > f` and
point this flag at it.

Exit status is 0 when no run is a proper prefix of a source string, and 1 when
one is. A construct the script cannot read - the rows' initialiser, a row that
is not five members, a `source` member that is not a string literal - is an
error naming the construct, never a quiet pass.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from dataclasses import dataclass

REPO = pathlib.Path(__file__).resolve().parent.parent

# The file the source strings are read from and the documents the runs are read
# from: the published set a reader outside this repository can open.
SOURCE = REPO / "src" / "boys.cpp"
DOCUMENTS = (
    REPO / "README.md",
    REPO / "docs" / "getting-started.md",
    REPO / "docs" / "lane-contract.md",
)

# The rows of BoysLaneContracts(): `std::array<LaneContractInfo, N> name = {{ ... }};`.
ROWS = re.compile(
    r"std::array<\s*LaneContractInfo\s*,\s*(?P<count>\d+)\s*>\s*"
    r"[A-Za-z_][A-Za-z0-9_]*\s*=\s*\{\{(?P<rows>.*?)\}\};",
    re.S,
)

# The members of a LaneContractInfo row: precision, name, bound, additive,
# source. The last is the one this check reads, and a revision that adds a
# field is a construct the readers below name rather than a row read short.
FIELDS = 6

# One or more adjacent string literals - which is how a value longer than a
# line is written here - and the escape sequences decoded below.
LITERALS = re.compile(r'"(?:[^"\\]|\\.)*"')
LITERAL_RUN = re.compile(r'(?:\s*"(?:[^"\\]|\\.)*")+\s*')
ESCAPES = {"\\": "\\", '"': '"', "'": "'", "n": "\n", "t": "\t", "r": "\r"}

# The shapes a document quotes a run in. The bracket is what
# examples/06_what_it_guarantees.cpp prints a row's source in, `(from ...)` is
# what examples/01_one_order.cpp prints it in, and `bound: ...` heads the
# provenance cell of docs/lane-contract.md's table. A bracket run may span
# lines and a `bound:` run may not, which is why the two classes differ.
SHAPES = (
    ("bracket [ ... ]", re.compile(r"\[(?P<run>[^\[\]]+)\]")),
    ("(from ...)", re.compile(r"\(from (?P<run>[^()]+)\)")),
    ("bound: ...", re.compile(r"bound: (?P<run>[^\n]+)")),
)

# The document's own marker inside a provenance cell: everything after `bound: `
# up to it is the run, and the text after it is the delivered figure's own
# provenance, which this check does not read.
CELL_END = "delivered:"

# The shortest run this check reads as a quotation rather than as a word that
# happens to start the same way.
MIN_RUN = 12

# How much of a source string's continuation a finding prints. The point is to
# show the reader where the field writes on, not to reproduce the field.
CONTINUATION = 110


class CheckError(Exception):
    """A construct the script cannot read. Never a quiet pass."""


@dataclass(frozen=True)
class SourceString:
    """One lane row's `source` member, with the name the row prints it under."""

    lane: str
    text: str


@dataclass(frozen=True)
class Run:
    """One run a document quotes, whitespace-collapsed, where it was read."""

    document: str
    line: int
    shape: str
    text: str


def display(path: pathlib.Path) -> str:
    """A path as the repository names it, so a report's line reads like a link."""
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


def read_text(path: pathlib.Path) -> str:
    """The file's text, with a decode failure named rather than replaced."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise CheckError(f"{path.name}: not UTF-8 at byte {error.start}: {error.reason}") from error


def strip_comments(text: str) -> str:
    """Remove C++ comments, keeping string and character literals intact.

    The rows are surrounded by prose that names the same things they do - the
    members, the lanes, the strings - so a comment left in place would be read
    as a row. Literals are kept byte for byte, because they are the subject.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == '"' or c == "'":
            quote = c
            out.append(c)
            i += 1
            while i < n:
                if text[i] == "\\" and i + 1 < n:
                    out.append(text[i : i + 2])
                    i += 2
                    continue
                out.append(text[i])
                if text[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def split_members(text: str, separator: str) -> list[str]:
    """Split at `separator` where no bracket is open, keeping literals whole."""
    pieces: list[str] = []
    current: list[str] = []
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == '"' or c == "'":
            quote = c
            current.append(c)
            i += 1
            while i < n:
                if text[i] == "\\" and i + 1 < n:
                    current.append(text[i : i + 2])
                    i += 2
                    continue
                current.append(text[i])
                if text[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == separator and depth == 0:
            pieces.append("".join(current))
            current = []
            i += 1
            continue
        current.append(c)
        i += 1
    pieces.append("".join(current))
    return [piece.strip() for piece in pieces]


def decode_literal(raw: str, where: str) -> str:
    """The characters a string literal stands for, or an error naming the escape."""
    body = raw[1:-1]
    out: list[str] = []
    i = 0
    while i < len(body):
        if body[i] != "\\":
            out.append(body[i])
            i += 1
            continue
        if i + 1 >= len(body) or body[i + 1] not in ESCAPES:
            raise CheckError(
                f"{where}: the escape {body[i : i + 2]!r} is not one this script decodes "
                f"(it reads {', '.join(sorted(repr(k) for k in ESCAPES))})"
            )
        out.append(ESCAPES[body[i + 1]])
        i += 2
    return "".join(out)


def join_literals(member: str, where: str) -> str:
    """The value a run of adjacent string literals spells, as one string."""
    if not member:
        raise CheckError(f"{where}: the value is empty")
    if LITERAL_RUN.fullmatch(member) is None:
        raise CheckError(
            f"{where}: the value is not a run of string literals ({member[:60]!r} ...); this "
            f"script reads a row's `source` member there and cannot read a value computed from "
            f"anything else"
        )
    return "".join(
        decode_literal(match.group(0), where) for match in LITERALS.finditer(member)
    )


def read_source_strings(path: pathlib.Path) -> list[SourceString]:
    """The lane rows' `source` members, or an error naming what could not be read."""
    if not path.is_file():
        raise CheckError(f"no such source file: {path}")

    text = strip_comments(read_text(path))
    match = ROWS.search(text)
    if match is None:
        raise CheckError(
            f"{path.name}: no `std::array<LaneContractInfo, N> name = {{{{ ... }}}};` initialiser "
            f"was found. This script reads the lane rows' `source` members there, so a revision "
            f"that builds them another way has to be told to it rather than read as no strings"
        )

    declared = int(match.group("count"))
    rows = [row for row in split_members(match.group("rows"), ",") if row]
    if len(rows) != declared:
        raise CheckError(
            f"{path.name}: the initialiser declares {declared} rows and holds {len(rows)}"
        )

    strings: list[SourceString] = []
    for index, row in enumerate(rows, 1):
        where = f"{path.name}: row {index}"
        if not (row.startswith("{") and row.endswith("}")):
            raise CheckError(f"{where}: not a braced initialiser ({row[:60]!r} ...)")
        members = split_members(row[1:-1], ",")
        if len(members) != FIELDS:
            raise CheckError(
                f"{where}: {len(members)} members, and LaneContractInfo has {FIELDS} "
                f"(precision, name, bound, additive, plainAdditive, source); the last is the one read here"
            )
        lane = " ".join(join_literals(members[1], where).split()) or f"row {index}"
        source = " ".join(join_literals(members[FIELDS - 1], where).split())
        strings.append(SourceString(lane, source))

    if not strings:
        raise CheckError(f"{path.name}: the initialiser holds rows and no row carries a string")
    return strings


def read_runs(path: pathlib.Path) -> list[Run]:
    """Every run of the three shapes the document marks as quoted."""
    text = read_text(path)
    document = display(path)
    runs: list[Run] = []
    for shape, pattern in SHAPES:
        for match in pattern.finditer(text):
            raw = match.group("run")
            if shape == "bound: ...":
                raw = raw.split(CELL_END)[0]
                raw = raw.rstrip().rstrip("|").rstrip()
            collapsed = " ".join(raw.split())
            if not collapsed:
                continue
            runs.append(
                Run(document, text.count("\n", 0, match.start("run")) + 1, shape, collapsed)
            )
    return runs


def spellings(run: Run) -> list[str]:
    """The run as quoted, and the same run with the document's full stop dropped.

    A `bound: ...` cell ends its sentence with a period the field does not have,
    and a source string may itself end with one; reading both spellings is what
    keeps the document's punctuation from being read as a missing character.
    """
    if run.text.endswith(".") and len(run.text) > 1:
        return [run.text, run.text[:-1].rstrip()]
    return [run.text]


def continuation(run_text: str, source: SourceString) -> str:
    """What the field says after the run, capped for the finding's own length."""
    rest = source.text[len(run_text) :]
    if len(rest) <= CONTINUATION:
        return rest
    return rest[:CONTINUATION].rstrip() + " ..."


def classify(runs: list[Run], strings: list[SourceString]) -> tuple[list, list, list, list, list]:
    """Sort the runs into mismatches, confirmed quotes, and the rest.

    A run that is both a proper prefix of one string and another string in full
    is a mismatch: the run names one lane's figure in a document that sits
    beside a row whose words have grown, and clearing it on the second reading
    would be the check answering a question it was not asked.
    """
    mismatches: list[tuple[Run, str, list[SourceString], list[SourceString]]] = []
    confirmed: list[tuple[Run, str, list[SourceString]]] = []
    unrelated: list[Run] = []
    short: list[tuple[Run, str, list[SourceString]]] = []
    beyond: list[tuple[Run, str, list[SourceString]]] = []

    for run in runs:
        texts = spellings(run)
        prefixes = [
            (text, source)
            for text in texts
            for source in strings
            if len(text) >= MIN_RUN and text != source.text and source.text.startswith(text)
        ]
        equals = [(text, source) for text in texts for source in strings if text == source.text]
        if prefixes:
            mismatches.append(
                (run, prefixes[0][0], [hit[1] for hit in prefixes], [hit[1] for hit in equals])
            )
            continue
        if equals:
            confirmed.append((run, equals[0][0], [e[1] for e in equals]))
            continue
        # Only a run with no exact match is asked the two questions the rule
        # above cannot answer about it. A run that is one row's string in full
        # is accounted for, even where another row's shorter string is the
        # whole of its beginning: that second reading is a row that says less,
        # and reading it as a finding here would report the documents' correct
        # quotations every time two rows share an opening clause.
        unrelated.append(run)
        short.extend(
            (run, text, source)
            for text in texts
            for source in strings
            if len(text) < MIN_RUN and text != source.text and source.text.startswith(text)
        )
        beyond.extend(
            (run, text, source)
            for text in texts
            for source in strings
            if text.startswith(source.text)
            and text != source.text
            and text.rstrip(".") != source.text
        )
    return mismatches, confirmed, unrelated, short, beyond


def report_strings(strings: list[SourceString], path: pathlib.Path, out) -> None:
    print(f"{display(path)}: the lane rows' `source` strings, {len(strings)} read", file=out)
    for source in strings:
        head = source.text if len(source.text) <= 64 else source.text[:61] + "..."
        print(f'  {source.lane:<12} {len(source.text):>5} characters  "{head}"', file=out)


def report_runs(runs: list[Run], out) -> None:
    """The runs read, per document and by shape.

    Printed per document rather than as one total: a document whose reader stops
    marking runs is a document this check no longer reads, and the count beside
    its name is where that shows.
    """
    print(f"\nruns read, by document ({len(runs)} in all)", file=out)
    for document in (display(document) for document in DOCUMENTS):
        read = [run for run in runs if run.document == document]
        by_shape = ", ".join(
            f"{shape} {sum(1 for run in read if run.shape == shape)}" for shape, _ in SHAPES
        )
        print(f"  {document:<24} {len(read):>4}  {by_shape}", file=out)
    for shape, _ in SHAPES:
        if not any(run.shape == shape for run in runs):
            print(
                f"  the shape {shape} read nothing at all: the documents have stopped marking a "
                f"run that way, or this reader no longer reads the marker",
                file=out,
            )


def report_clean(confirmed, unrelated, short, beyond, strings, path, out) -> None:
    print(f"\nconfirmed: {len(confirmed)} runs are a source string in full", file=out)
    for run, text, matches in confirmed:
        lanes = " and ".join(dict.fromkeys(source.lane for source in matches))
        print(f'  {run.document}:{run.line} {run.shape:<16} {lanes}  "{text}"', file=out)

    print(f"\nnot the field's text at all: {len(unrelated)} runs", file=out)
    print(
        "  the documents' cross-reference labels, their shell text and their own prose: a run of "
        "those is read here and nothing is claimed about it",
        file=out,
    )

    if short:
        print(f"\nshorter than a sentence, so not read as a quotation: {len(short)}", file=out)
        for run, text, source in short:
            print(f'  {run.document}:{run.line}: "{text}" begins the {source.lane} row', file=out)

    if beyond:
        print(
            f"\nlonger than the string it begins with, which this check cannot decide: "
            f"{len(beyond)}",
            file=out,
        )
        print(
            "  a paraphrase and a quotation of a field that has since been shortened read the same "
            "here, so these are printed and never failed",
            file=out,
        )
        for run, text, source in beyond:
            print(
                f'  {run.document}:{run.line}: "{text[:96]}" contains the whole {source.lane} row',
                file=out,
            )

    print(
        f"\nverdict: clean - no run in these documents is a proper prefix of a source string "
        f"({len(confirmed)} confirmed, {len(unrelated)} unconfirmed, {len(strings)} strings read "
        f"from {display(path)})",
        file=out,
    )


def report_mismatches(mismatches, runs, strings, path, out) -> None:
    """The mismatches, by the run rather than by the place it was read.

    One quoted sentence is usually quoted in several places, and a finding that
    repeats it once per place buries the two texts under its own locations.
    """
    groups: dict[tuple, dict] = {}
    for run, text, sources, equals in mismatches:
        key = (text, tuple(hit.lane for hit in sources), tuple(hit.lane for hit in equals))
        group = groups.setdefault(
            key, {"text": text, "sources": sources, "equals": equals, "runs": []}
        )
        group["runs"].append(run)

    print(
        f"\n{len(mismatches)} run(s) are a proper prefix of a source string, "
        f"in {len(groups)} distinct finding(s):",
        file=out,
    )
    for group in groups.values():
        text = group["text"]
        places = ", ".join(f"{run.document}:{run.line} ({run.shape})" for run in group["runs"])
        print(f'\nMISMATCH: the run "{text}", read at {len(group["runs"])} place(s)', file=out)
        print(f"  {places}", file=out)
        for source in group["sources"]:
            print(
                f"  the {source.lane} row's `source` is {len(source.text)} characters long and "
                f"this run is its first {len(text)}: it does not stop where the run does",
                file=out,
            )
            print(f'    "{continuation(text, source)}"', file=out)
        if group["equals"]:
            lanes = " and ".join(dict.fromkeys(source.lane for source in group["equals"]))
            print(
                f"  the run is the {lanes} row's `source` in full, which is why it is reported "
                f"rather than cleared: it matches one row's string in full and stops short of "
                f"another's, and a document sitting beside both has to say which it means",
                file=out,
            )
    print(
        f"\nA document that quotes this field is quoting what the programs print, so a run that "
        f"stops where the field writes on describes output no program produces. Either carry the "
        f"field in full or take the run out of the shape the reader reads it in.",
        file=out,
    )
    print(
        f"\n{len(mismatches)} of {len(runs)} runs read, against {len(strings)} strings from "
        f"{display(path)}.",
        file=out,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="the form the other tools take: report and exit non-zero on a difference",
    )
    parser.add_argument(
        "--source",
        default=str(SOURCE),
        help="the translation unit to read the lane rows from (default: src/boys.cpp)",
    )
    args = parser.parse_args()

    try:
        path = pathlib.Path(args.source).resolve()
        strings = read_source_strings(path)
        missing = [document for document in DOCUMENTS if not document.is_file()]
        if missing:
            raise CheckError(
                f"no such document: {', '.join(str(document) for document in missing)}; this check "
                f"reads the runs these documents quote"
            )
        runs = [run for document in DOCUMENTS for run in read_runs(document)]
        mismatches, confirmed, unrelated, short, beyond = classify(runs, strings)
        if not confirmed and not mismatches:
            by_shape = ", ".join(
                f"{shape} {sum(1 for run in runs if run.shape == shape)}" for shape, _ in SHAPES
            )
            raise CheckError(
                f"no run in these documents could be confirmed against a source string: this "
                f"check has no subject. {len(runs)} runs were read ({by_shape}) and none of them "
                f"equals a `source` member of {display(path)}'s lane rows. Either the documents "
                f"stopped quoting the field or this reader no longer reads them"
            )
    except CheckError as error:
        print(f"check_doc_transcripts: {error}", file=sys.stderr)
        return 1

    out = sys.stdout
    print(
        "check_doc_transcripts: the documents' quoted runs against the library's source strings\n"
    )
    report_strings(strings, path, out)
    report_runs(runs, out)
    if mismatches:
        report_mismatches(mismatches, runs, strings, path, out)
        return 1
    report_clean(confirmed, unrelated, short, beyond, strings, path, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
