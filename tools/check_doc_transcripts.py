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
every run of those shapes out of README.md, docs/getting-started.md and
docs/lane-contract.md, and compares them after collapsing each side's
whitespace - a document may wrap a run across lines and the field has none, and
a line break is a document's, not a quotation's. It then reports four kinds of
run:

  * a run that is a proper prefix of a source string and at least twelve
    characters long is a MISMATCH, and it fails the run. That is the shape the
    drift took: the document carries the text it once quoted and the field
    writes on past it. The finding names the row whose string the run begins,
    how many of that string's characters it carries, and what the field says
    next, so the reader has the two texts rather than a verdict about them.

  * a run equal to a source string is a confirmed quote, named and counted.

  * a run the document marks as the library's own words, and which no string the
    library holds carries, is ABSENT, and it fails the run. The prefix rule
    above cannot see this one: a string the source has deleted altogether has no
    proper prefix to be a prefix of, and a run of it is a run of nothing. The
    strings the library holds are read from `src/` and `include/` - every string
    literal the published library carries, and not the lane rows' `source`
    members alone, because the sentence a refusal cell prints is the accessor's
    and is written wherever the accessor is.

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

Which runs are marked as the library's own words, and why the two markings are
not read the same way. A `bound: ...` cell and the italic run of a refusal cell
both print the library's words and say so. The refusal cell prints the sentence
a call gets, as the accessor's own, so it is a transcript and not a restatement:
a string holds those characters or the cell is describing a refusal this
revision does not make, and the run is ABSENT. The `bound: ...` cells are
restatements, and the documents have always written them that way. The fp16 cell
has said "value, claimed" since the commit that wrote the row, where the field
says "value and claimed"; the fp32-device cell opens with the document's own
gloss - "the lane's documented figure, " - before the field's words begin.
Neither is a drift and neither is an error, and a rule that read them as
transcripts would report those cells on every run. So the rule there is the
weakest one that still sees a deletion: a `bound: ...` run that agrees with no
lane row's `source` on a run of at least twelve characters is ABSENT, and a run
that agrees on any run at all is passed. What that weaker rule misses is a
`bound: ...` cell rewritten into words that keep a dozen characters of a row's
old string, and one edited into a sentence no row ever carried while still
sharing a phrase with the row it names; both are read as the document's prose,
which is what the two texts alone make of them.

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
read as the table's own prose - with the one exception above, a run that is no
part of any row's field at all. The refusal cell is compared against the
library's strings instead, because what it prints is the accessor's sentence
rather than a lane's terms, and no `source` member carries it.

What this check does not read, and why:

  * the `delivered: ...` half of those cells, whose runs carry markdown
    backticks the field does not have, and the delivery accessor's `source`
    strings behind them, which name the tables a figure was read from rather
    than terms the base does not carry and reach no document in one of the
    shapes above;

  * `examples/`, whose output is this check's subject rather than an input;

  * the figures the same lines print. `tools/check_doc_arithmetic.py` reads the
    arithmetic of the prose and the accuracy gate reads the bounds; a figure
    beside a run is neither this check's subject nor one it can confirm;

  * a marked run the library still holds *in part*. A cell cut down to the first
    half of a sentence the library still has is a MISMATCH where the run is a
    proper prefix of a row's `source`, and is nothing this check can name
    anywhere else: a run that stops inside a string the library holds reads the
    same as a run of the document's own words, and the two texts do not tell
    them apart.

Usage:
    python tools/check_doc_transcripts.py --check
    python tools/check_doc_transcripts.py --check --source /tmp/old_boys.cpp

`--source` reads the lane rows from another file, which is how the
pre-restoration tree is reproduced as a negative control: `git show
<rev>:src/boys.cpp > f` and point this flag at it. The flag stands in for the
file and not for the tree, and the library's strings are read from `src/` and
`include/` with that file in the place of `src/boys.cpp`, so the same document
can be asked twice: against the revision whose strings it was written beside,
and against the tree's own. The difference between the two runs is the drift,
measured rather than asserted.

Exit status is 0 when no run is a proper prefix of a source string and no marked
run is absent from the strings the library holds, and 1 when one is. A construct
the script cannot read - the rows' initialiser, a row that is not six members, a
`source` member that is not a string literal, a `src/` or `include/` that is not
there - is an error naming the construct, never a quiet pass.
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

# The published library whose strings a marked run has to be one of. The lane
# rows' `source` members are read from SOURCE; the sentence a refusal cell
# prints is the accessor's own and is written wherever the accessor is, which is
# why these are the two trees and not one file.
LIBRARY = (REPO / "src", REPO / "include")
LIBRARY_SUFFIXES = (".cpp", ".hpp", ".h", ".cu")

# The rows of BoysLaneContracts(): `std::array<LaneContractInfo, N> name = {{ ... }};`.
ROWS = re.compile(
    r"std::array<\s*LaneContractInfo\s*,\s*(?P<count>\d+)\s*>\s*"
    r"[A-Za-z_][A-Za-z0-9_]*\s*=\s*\{\{(?P<rows>.*?)\}\};",
    re.S,
)

# The members of a LaneContractInfo row this check reads: precision, name, bound,
# additive, plainAdditive, source. The last of them is the one whose string this
# check reads. A row may leave the members after it to their defaults - the plain
# reciprocal's own sentence is one - so a row carries at least these and at most
# the struct's own count, and a revision that adds a field the row has to write
# before `source` is a construct the readers below name rather than a row read
# short.
FIELDS = 6

# The members LaneContractInfo declares: the six above and the plain
# reciprocal's own sentence, which trails `source` and which no row has to write.
STRUCT_FIELDS = 7

# One or more adjacent string literals - which is how a value longer than a
# line is written here - and the escape sequences decoded below.
LITERALS = re.compile(r'"(?:[^"\\]|\\.)*"')
LITERAL_RUN = re.compile(r'(?:\s*"(?:[^"\\]|\\.)*")+\s*')
ESCAPES = {"\\": "\\", '"': '"', "'": "'", "n": "\n", "t": "\t", "r": "\r"}

# The shapes a document quotes a run in. The bracket is what
# examples/06_what_it_guarantees.cpp prints a row's source in, `(from ...)` is
# what examples/01_one_order.cpp prints it in, `bound: ...` heads the
# provenance cell of docs/lane-contract.md's table, and the italic run after
# `no verdict, no figure:` is the sentence docs/lane-contract.md prints as the
# accessor's own. A bracket run may span lines and the other three may not,
# which is why the classes differ.
SHAPES = (
    ("bracket [ ... ]", re.compile(r"\[(?P<run>[^\[\]]+)\]")),
    ("(from ...)", re.compile(r"\(from (?P<run>[^()]+)\)")),
    ("bound: ...", re.compile(r"bound: (?P<run>[^\n]+)")),
    ("refusal * ... *", re.compile(r"no verdict, no figure: \*(?P<run>[^*\n]+)\*")),
)

# The two shapes that mark a run as the library's own words rather than as the
# document's own prose. `bound: ...` restates a lane row's `source` in a figure's
# provenance cell, and a refusal cell's italic run prints the sentence a call
# gets. The names are the ones SHAPES carries, because a rule is chosen by the
# shape it was read in.
BOUND_SHAPE = "bound: ..."
REFUSAL_SHAPE = "refusal * ... *"

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


@dataclass(frozen=True)
class Absent:
    """One marked run that no string the library holds carries.

    `best` and `lane` are what a `bound: ...` cell came closest to: the longest
    run of characters it shares with a lane row's `source`, and the row it
    shares it with. They are what tells the reader the cell restates no row. A
    refusal cell is read as the accessor's transcript, so there is nothing
    closer for it to be and the two stay at their zeros.
    """

    run: Run
    text: str
    kind: str
    best: int = 0
    lane: str = ""


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
        if not FIELDS <= len(members) <= STRUCT_FIELDS:
            raise CheckError(
                f"{where}: {len(members)} members, and a row carries {FIELDS} to {STRUCT_FIELDS} "
                f"of LaneContractInfo's (precision, name, bound, additive, plainAdditive, source, "
                f"plainSource); `source`, the last of the six this check reads, is the one it "
                f"reads here"
            )
        lane = " ".join(join_literals(members[1], where).split()) or f"row {index}"
        source = " ".join(join_literals(members[FIELDS - 1], where).split())
        strings.append(SourceString(lane, source))

    if not strings:
        raise CheckError(f"{path.name}: the initialiser holds rows and no row carries a string")
    return strings


def library_files(source: pathlib.Path) -> list[pathlib.Path]:
    """Every file the published library's strings are read from.

    `src/` and `include/`, with `source` standing in for the file it names.
    `--source` stands in for the lane rows' translation unit, and the accessor's
    sentences are written in that same file, so naming another revision of it
    reproduces that revision for both - which is what lets the drift this check
    reads be shown as a difference between two runs of it rather than asserted.
    """
    files: list[pathlib.Path] = []
    replaced = SOURCE.resolve()
    for root in LIBRARY:
        if not root.is_dir():
            raise CheckError(
                f"no such directory: {display(root)}; this check reads the strings the published "
                f"library holds - the accessor's own sentences among them - out of src/ and "
                f"include/, so a tree without one has no strings to read"
            )
        files.extend(
            path
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and path.suffix in LIBRARY_SUFFIXES
            and path.resolve() != replaced
        )
    if not source.is_file():
        raise CheckError(f"no such source file: {source}")
    files.append(source)
    return files


def read_library_strings(files: list[pathlib.Path]) -> list[str]:
    """Every string literal the published library holds, whitespace-collapsed.

    One string per literal, or per run of adjacent literals, because C++ joins
    those and a sentence longer than a line is written that way here. Comments
    go and literals stay whole: what a document quotes is the characters a
    program holds, not the file's layout around them. A literal with an escape
    this script does not decode is an error naming it, and a library that holds
    no string at all is an error too, because a marked run cannot be held to a
    library with nothing to hold it to.
    """
    strings: list[str] = []
    seen: set[str] = set()
    for path in files:
        text = strip_comments(read_text(path))
        for match in LITERAL_RUN.finditer(text):
            where = f"{display(path)}: offset {match.start()}"
            value = " ".join(join_literals(match.group(0), where).split())
            if value and value not in seen:
                seen.add(value)
                strings.append(value)
    if not strings:
        raise CheckError(
            f"no string literal was read from {', '.join(display(path) for path in files)}: a "
            f"marked run cannot be held to a library that has no strings"
        )
    return strings


def longest_shared_run(text: str, other: str) -> int:
    """The longest run of characters the two texts have in common.

    The unit a `bound: ...` cell is held to. A restatement keeps runs of the
    row's words - the fp16 cell carries sixty-three characters of its row's
    `source` - and a cell written in words no row ever carried keeps none of
    them.
    """
    previous = [0] * (len(other) + 1)
    best = 0
    for character in text:
        current = [0] * (len(other) + 1)
        for index in range(1, len(other) + 1):
            if character == other[index - 1]:
                current[index] = previous[index - 1] + 1
                if current[index] > best:
                    best = current[index]
        previous = current
    return best


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

    A cell that marks a run ends its sentence with a period the string it quotes
    does not have, and a source string may itself end with one; reading both
    spellings is what keeps the document's punctuation from being read as a
    missing character.
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


def classify(
    runs: list[Run], strings: list[SourceString], sentences: list[str]
) -> tuple[list, list, list, list, list, list]:
    """Sort the runs into mismatches, absent marked runs, confirmed quotes, and the rest.

    A run that is both a proper prefix of one string and another string in full
    is a mismatch: the run names one lane's figure in a document that sits
    beside a row whose words have grown, and clearing it on the second reading
    would be the check answering a question it was not asked.

    A run is held to the library's strings only where the document marks it as
    the library's words, and the two markings are held differently. A refusal
    cell prints the accessor's sentence as the accessor's own, so it is a
    transcript: a string holds it or the cell is quoting a refusal this revision
    does not make. A `bound: ...` cell is a restatement - the document adds its
    own connectives and its own gloss around the row's words - so it is held to
    the weakest rule that still sees a deletion, an agreement of at least
    MIN_RUN characters with some row's `source`, and only a cell that agrees
    with no row on that much is absent.
    """
    mismatches: list[tuple[Run, str, list[SourceString], list[SourceString]]] = []
    absent: list[Absent] = []
    confirmed: list[tuple[Run, str, list[SourceString]]] = []
    unrelated: list[Run] = []
    short: list[tuple[Run, str, list[SourceString]]] = []
    beyond: list[tuple[Run, str, list[SourceString]]] = []

    for run in runs:
        texts = spellings(run)
        if run.shape == REFUSAL_SHAPE:
            if not any(text in sentence for text in texts for sentence in sentences):
                absent.append(Absent(run, texts[0], "refusal"))
            continue
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
        if run.shape == BOUND_SHAPE:
            agreements = [
                (longest_shared_run(text, source.text), source)
                for text in texts
                for source in strings
            ]
            best, source = max(agreements, key=lambda pair: pair[0])
            # A run shorter than MIN_RUN cannot hold an agreement that long, and
            # this check reads a shorter run as a word rather than a quotation
            # wherever it meets one, so it is passed here as it is there.
            if len(texts[0]) >= MIN_RUN and best < MIN_RUN:
                absent.append(Absent(run, texts[0], "bound", best, source.lane))
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
    return mismatches, absent, confirmed, unrelated, short, beyond


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


def report_clean(confirmed, unrelated, short, beyond, strings, sentences, path, out) -> None:
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
        f"\nverdict: clean - no run in these documents is a proper prefix of a source string, and "
        f"every run they mark as the library's own words is one the library holds "
        f"({len(confirmed)} confirmed, {len(unrelated)} unconfirmed, {len(strings)} lane strings "
        f"read from {display(path)}, {len(sentences)} strings read from the library)",
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


def report_absent(absent, runs, strings, sentences, out) -> None:
    """The marked runs that no string the library holds carries.

    One finding per run rather than per place, for the reason the mismatches are
    grouped: the same words are quoted wherever the same refusal is described.
    The rule the run was read under is printed with it, because the two are not
    equally strong - a refusal cell is held to the accessor's own characters, and
    a `bound: ...` cell only to an agreement of MIN_RUN characters with the row
    it names - and a reader deciding what to do about the finding needs to know
    which of the two found it.
    """
    print(
        f"\n{len(absent)} run(s) the documents mark as the library's own words are absent from "
        f"every string it holds, in {len({finding.text for finding in absent})} distinct "
        f"finding(s):",
        file=out,
    )
    for finding in absent:
        run = finding.run
        print(
            f'\nABSENT: the run "{finding.text}", read at {run.document}:{run.line} ({run.shape})',
            file=out,
        )
        if finding.kind == "refusal":
            print(
                f"  the cell prints the sentence a call gets, as the accessor's own, and none of "
                f"the {len(sentences)} string literals the published library carries contains it: "
                f"a transcript of a refusal this revision does not make",
                file=out,
            )
            print(
                f"  a transcript is held to the characters rather than to a reading of them, "
                f"because the cell claims the accessor's own sentence: the library says these "
                f"words or the cell is describing output it no longer produces",
                file=out,
            )
        else:
            print(
                f"  the cell names the row's bound, and the closest lane row - the "
                f"{finding.lane} one - agrees with it on {finding.best} character(s): a cell that "
                f"shares no run of {MIN_RUN} characters with any of the {len(strings)} rows' "
                f"`source` strings restates none of them",
                file=out,
            )
            print(
                f"  this is the weaker of the two rules the marked shapes are read under, and "
                f"what it cannot tell apart is a string that has left the library from a cell "
                f"written in words no row ever carried: both agree with every row on nothing "
                f"worth reading, and the two texts do not separate them",
                file=out,
            )
    print(
        f"\nA run in one of these shapes is the document's claim that these are the library's "
        f"characters, and this check reads it as that claim. Either carry the string the library "
        f"holds today or take the run out of the shape that marks it as one.",
        file=out,
    )
    print(
        f"\n{len(absent)} of {len(runs)} runs read, against {len(strings)} row strings and "
        f"{len(sentences)} library strings.",
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
        sentences = read_library_strings(library_files(path))
        missing = [document for document in DOCUMENTS if not document.is_file()]
        if missing:
            raise CheckError(
                f"no such document: {', '.join(str(document) for document in missing)}; this check "
                f"reads the runs these documents quote"
            )
        runs = [run for document in DOCUMENTS for run in read_runs(document)]
        mismatches, absent, confirmed, unrelated, short, beyond = classify(runs, strings, sentences)
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
    if mismatches or absent:
        if mismatches:
            report_mismatches(mismatches, runs, strings, path, out)
        if absent:
            report_absent(absent, runs, strings, sentences, out)
        return 1
    report_clean(confirmed, unrelated, short, beyond, strings, sentences, path, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
