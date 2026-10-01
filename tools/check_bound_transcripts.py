#!/usr/bin/env python3
"""Check the bound figures the accuracy gate transcribes against the documents that state them.

`tests/boys_accuracy_gate.cpp` judges every lane and region of this library against a bound, and
those bounds are constants in the gate: `kBoundSingleA`, `kBoundSingleBand`, `kBoundSingleB`,
`kBoundSingleC`, `kBoundDoubleBatch`, `kBoundFloat`, `kNativeUlpBound`. The block they are declared
in says where they come from - "transcribed from README.md and include/boys/boys.hpp" - and until
this script nothing read the two against each other. A figure edited in a document and not in the
gate leaves the gate certifying its own transcription: the run is green, the claim printed beside it
is the document's, and the number under it is the gate's.

This script reads the figures off both sides and compares them. The sides are:

  * the gate's constants, read from `tests/boys_accuracy_gate.cpp`;
  * `include/boys/boys.hpp`, whose preamble carries the contract table - one cell per lane and per
    region, which is the finest statement of these figures in the tree;
  * `README.md`, two tables of it: the contract table under "Accuracy contract", which resolves by
    region in its prose, and the table of the gate's own printed run, whose `Bound claimed` column
    is the gate's row identity and its bound;
  * `src/boys.cpp`, whose `BoysLaneContracts()` rows are the library's own statement of one figure
    per lane plus an additive term - the side the documents are a promise to.

THE CORRESPONDENCE IS NOT ONE TO ONE, and this check does not pretend it is. The header's table has
a cell per lane and region; the README's contract table has one cell per lane holding every figure
that lane publishes, with the regions resolved in prose; and `BoysLaneContracts()` holds one figure
per lane. So each tie below declares, for one gate constant, the header cell it must equal, the
README contract row whose figures it must be one of, and - where the README's printed-run table has
a row for the same lane and region - that cell. Where several constants come off one row (the double
single lane's four regions off the README's one row), the row's figures are compared as a **set**:
the row states three figures and the gate four, and which of the row's figures belongs to which of
its thresholds is the row's own prose. Reading that prose as a partition would be a mapping this
check invented, so it compares the sets and says so here.

A figure the check cannot tie is named rather than passed over. Two lists of them are declared in
this file, each entry with the reason it is there:

  * `UNTIED` - a constant of the gate that holds a documented figure and is not the same fact as any
    document's cell, with why not. `kWithdrawnHeaderABound` is the one on this revision: it is the
    gate's record of a bound the header has stopped claiming, so there is no cell to hold it to.
    What the check can read about it is read: the figure is required to be absent from the header's
    double-single row outside region A, because a header that claimed it there again would make the
    gate's own note - and the finding it keeps re-measurable - wrong about the file it describes.
  * `UNCOVERED` - a figure-bearing row of a document that no tie holds, with why not.

A row that carries a figure and is in neither list is an error naming the row, not a quiet pass, and
so is a constant of the gate that holds a documented figure and is declared nowhere: the drift this
check exists for is exactly a new transcription nobody tied. Where the reader cannot read a
construct at all - a table whose header has moved, a cell with no figure in it, a lane row whose
members are not five - it names the construct and exits non-zero. Zero figures read from the
documents is an error: a check that can pass by reading nothing is not a check.

What this check does not read, and why:

  * the widths, thresholds and boundaries in those tables - `x = 11.899848152108484`,
    `x = 28.984375`, `x = 1.0855`. They are the documents' own coordinates, no constant of the gate
    states them as a bound figure, and the gate's own region edges are measured elsewhere. A
    threshold edited in one document and not another is not a figure this check can see;
  * the `Worst delivered` column of the README's printed-run table, and every other delivered
    figure in the prose. A delivered figure is a run's own output - the gate prints it and the
    gate's claims are what hold it - while this check reads the figures a document *states*;
  * the pair of CUDA rows of the README's contract table whose cells are cross-references ("same
    m-budgets as the CPU double lanes") and carry no figure at all;
  * the half-ULP figures of the fp16/bf16 rows. The README's printed-run table names them as the
    numbers a run found (see UNCOVERED); the header and the README's contract table state that term
    as prose ("+ 1/2 ULP") with no number in it, and the gate composes it at run time from the
    format's quantum at the returned value rather than from a constant;
  * the ULP figure 8 in the sweep below, which is a literal any file may carry for its own reasons.
    The constant that states it, `kNativeUlpBound`, is tied by name like the rest;
  * the same figures where the pages under `docs/` restate them - `docs/lane-contract.md`'s prose
    and its per-route tables, and the program output `docs/getting-started.md` quotes. Those pages
    state a lane's figure beside figures of their own (a route's measured bar, a rung's budget, an
    example's run), and holding the gate to them would need a mapping from those cells to the lane's
    that this check does not have. A page that drifts from the gate is a gap this tool does not
    close, and this list is where that is said rather than left to be assumed.

Usage:
    python tools/check_bound_transcripts.py --check
    python tools/check_bound_transcripts.py --check --readme /tmp/README.md

`--gate`, `--header`, `--readme` and `--library` read a side from another file, which is how a
shifted figure is shown to fail: copy the file, edit the copy, point the flag at it. No document and
no source is written by this script.

Exit status is 0 when every tied figure agrees, and 1 when one does not, when a constant holding a
documented figure is declared nowhere, when a figure-bearing row is held by no tie, or when a
construct could not be read.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from dataclasses import dataclass

REPO = pathlib.Path(__file__).resolve().parent.parent
GATE = REPO / "tests" / "boys_accuracy_gate.cpp"
HEADER = REPO / "include" / "boys" / "boys.hpp"
README = REPO / "README.md"
LIBRARY = REPO / "src" / "boys.cpp"

# The middle dot the documents write a figure after, kept as an escape so this
# file is ASCII: `m<dot>5.5e-14` is how both documents state a multiplied bound.
MULT = re.compile(r"m[\u00b7*]\s*(?P<num>\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
# The additive form, `m<dot>1.5e-7 + 8e-8`, which the README's fast device row
# and the library's own additive member both state. A trailing `+ 1/2 ULP`
# carries no number and is prose, not a figure.
ADDEND = re.compile(r"\s*\+\s*(?P<num>\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
ULP = re.compile(r"(?P<num>\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*ULP")
# A cell that is one number and nothing else, which is the shape the README's
# printed-run table writes its bound column in.
BARE = re.compile(r"^(?P<num>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)$")

# Anchored to one line: `\s` would let the match begin on the blank lines a
# stripped comment leaves behind, and the line a report names would be the first
# of those rather than the declaration's own.
DECLARATION = re.compile(
    r"^[ \t]*constexpr[ \t]+(?:double|float)[ \t]+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)[ \t]*=[ \t]*(?P<init>[^;]*);",
    re.M,
)
NUMBER = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$")

# The library's own lane rows, `std::array<LaneContractInfo, N> name = {{ ... }};`.
LANE_ROWS = re.compile(
    r"std::array<\s*LaneContractInfo\s*,\s*(?P<count>\d+)\s*>\s*"
    r"[A-Za-z_][A-Za-z0-9_]*\s*=\s*\{\{(?P<rows>.*?)\}\};",
    re.S,
)
# precision, name, bound, additive, source.
LANE_MEMBERS = 6

# The column of the header's contract table, by the label this check reads it
# under. Matching is by prefix so a parenthesised range beside a label - the
# header writes `region B (x0 <= x < x1)` - does not have to be transcribed here
# as well; two columns matching one prefix is an error, not a choice.
REGIONS = {"A": "region A", "band": "extended band", "B": "region B", "C": "region C"}


class CheckError(Exception):
    """A construct the script cannot read. Never a quiet pass."""


@dataclass(frozen=True)
class Figure:
    """One number read off one side, with where it was read."""

    value: float
    unit: str  # "absolute" | "additive" | "ulp"
    text: str
    where: str

    @property
    def key(self) -> tuple[float, str]:
        return (self.value, self.unit)


@dataclass(frozen=True)
class Row:
    """One table row: its cells, and the line it is on."""

    cells: tuple[str, ...]
    line: int

    @property
    def label(self) -> str:
        return self.cells[0]


@dataclass(frozen=True)
class Table:
    """A markdown or Doxygen table, read as its header cells and its rows."""

    source: str
    line: int
    columns: tuple[str, ...]
    rows: tuple[Row, ...]


@dataclass(frozen=True)
class Tie:
    """One gate constant, and the document cells that state the same fact."""

    constant: str
    header: str  # the header table's row label
    columns: tuple[str, ...]  # the header table's columns, by REGIONS key
    readme: str  # the README contract table's row label
    unit: str = "absolute"
    regions: tuple[tuple[str, str], ...] = ()  # (lane, region) in the README's run table


@dataclass(frozen=True)
class LibraryTie:
    """One `BoysLaneContracts()` row, and the document rows that state its figures."""

    row: str  # the lane name the library's row carries
    readme: str
    header: str | None = None


@dataclass(frozen=True)
class Untied:
    """A constant of the gate this check does not hold to a document, and why."""

    constant: str
    reason: str


@dataclass(frozen=True)
class Uncovered:
    """A figure-bearing row of a document this check does not hold, and why."""

    table: str  # "header" | "readme-contract" | "readme-run"
    label: str
    region: str
    reason: str


# ---------------------------------------------------------------------------
# The ties.
#
# Each is one figure that is the same fact on both sides, and the ground for
# saying so is the identity both sides already carry: a lane and a region in the
# header's own words, a lane row in the README's, and the gate's own claim label
# for the printed-run rows - `AddClaim("double single", "A", kBoundSingleA)` is
# the row the README prints as `double single | A`.
# ---------------------------------------------------------------------------
TIES = (
    Tie("kBoundSingleA", "double single", ("A",), "double single",
        regions=(("double single", "A"),)),
    Tie("kBoundSingleBand", "double single", ("band",), "double single",
        regions=(("double single", "band"),)),
    Tie("kBoundSingleB", "double single", ("B",), "double single",
        regions=(("double single", "B"),)),
    Tie("kBoundSingleC", "double single", ("C",), "double single",
        regions=(("double single", "C"),)),
    # The double batch lane's figure is flat over the whole domain, so all four
    # of the header's columns carry it and the README's contract row carries it
    # once. The README's printed run has no row for the lane.
    Tie("kBoundDoubleBatch", "double batch", ("A", "band", "B", "C"), "double batch"),
    Tie("kBoundFloat", "float single / batch", ("A", "band", "B", "C"), "float single / batch",
        regions=(("float single", "all"),)),
    # The native half lane's bound is in ULP of the returned value, not an
    # absolute figure: the number that ties is the 8, and the unit is carried
    # with it so it can never be compared against an absolute cell.
    Tie("kNativeUlpBound", "native half", ("C",), "native half", unit="ulp"),
)

# The gate's other statement of a tied figure, under a name of its own: the
# float lane's rung bar, which is that lane's published figure read for rung
# accounting. It is held to the same cells as the primary, so it cannot drift
# away from the document while the primary stays put.
ALIASES = (("kF32RungRegionBound", "kBoundFloat"),)

LIBRARY_TIES = (
    # The double batch entry is a double-precision entry: the README's row for
    # it is the row the library's fp64 lane states.
    LibraryTie("fp64", "double batch", "double batch"),
    LibraryTie("fp32", "float single / batch", "float single / batch"),
    LibraryTie("fp16", "fp16 / bf16", "fp16 / bf16"),
    # The device lane's additive, 8e-8, is stated by no constant of the gate at
    # all - it is here that it is read, against the README row that publishes
    # it. The header's table has no device row to hold it to.
    LibraryTie("fp32-device", "CUDA fp32, `RegionBExp::kFast`"),
)

UNTIED = (
    Untied(
        "kWithdrawnHeaderABound",
        "it is the gate's record of a bound the header has stopped claiming - a single region-A "
        "cell of 1e-15 over the whole of x < x0, which the gate found to be an over-claim and the "
        "header has since split into its region-A and extended-band cells. No cell of either "
        "document states it, and holding it to one would be holding the gate to a claim the "
        "documents have withdrawn. What can be read about it is read below: it must not reappear "
        "in the header's double-single row outside region A, or the gate's note would be wrong "
        "about the file it describes",
    ),
)

UNCOVERED = (
    Uncovered(
        "readme-run",
        "fp16 store-half",
        "single",
        "the cell holds the figure the gate's own run printed for the lane - the half-quantum of "
        "representation the store-half claim adds to the single-precision figure. The gate "
        "computes that term at run time from the format's quantum at the returned value, so no "
        "constant of it states the cell's number and the number moves with the run and with the "
        "argument the run found worst. This check holds constants against documents; where the "
        "term is composed is the gate's own claim for the lane, not a figure here",
    ),
    Uncovered(
        "readme-run",
        "bf16 store-half",
        "single",
        "as the fp16 row above: the half-quantum term of the same claim, computed at run time "
        "from bf16's own quantum at the returned value",
    ),
)


def display(path: pathlib.Path) -> str:
    """A path as the repository names it, so a report's line reads like a link."""
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


def read_text(path: pathlib.Path) -> str:
    """The file's text, with a decode failure named rather than replaced."""
    if not path.is_file():
        raise CheckError(f"no such file: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise CheckError(f"{path.name}: not UTF-8 at byte {error.start}: {error.reason}") from error


def ascii_safe(text: str) -> str:
    """A cell's text, printed on a console whose code page is not the file's.

    The documents are UTF-8 and carry a multiplication dot, a less-or-equal sign
    and a combining accent; a report that quoted them raw would fail to encode
    on a Windows console, which would not be a finding about the documents.
    """
    # The characters the documents carry that ASCII does not, written as
    # the escapes Python reads so this file stays ASCII.
    table = {
        "\u00b7": "*",
        "\u2264": "<=",
        "\u2265": ">=",
        "\u2014": "-",
        "\u2013": "-",
        "\u00bd": "1/2",
        "\u0302": "",
        "\u2212": "-",
        "\u00d7": "x",
    }
    out = "".join(table.get(character, character) for character in text)
    return out.encode("ascii", "backslashreplace").decode("ascii")


def strip_comments(text: str) -> str:
    """Remove C++ comments, keeping string and character literals intact.

    A commented-out declaration is not a declaration, and a comment between two
    lane rows would be read as a member of one.
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
                # The comment's newlines stay, because a report quotes the line
                # a declaration is on and the stripped text is what is counted.
                if text[i] == "\n":
                    out.append("\n")
                i += 1
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def table_cells(line: str) -> tuple[str, ...] | None:
    """The cells of one table line, or None when the line is not a table line.

    A pipe inside a cell is escaped in the header's table (`\\|F - F\\| <= ...`),
    so the split runs on unescaped pipes only.
    """
    rest = line
    for marker in ("///", "//!", "*/"):
        if rest.startswith(marker):
            rest = rest[len(marker) :]
            break
    rest = rest.lstrip()
    if rest.startswith("*"):
        rest = rest[1:].lstrip()
    if not rest.startswith("|"):
        return None
    body = rest.rstrip()
    if body.endswith("|"):
        body = body[:-1]
    body = body[1:]
    cells: list[str] = []
    current: list[str] = []
    i = 0
    while i < len(body):
        if body[i] == "\\" and i + 1 < len(body):
            current.append(body[i : i + 2])
            i += 2
            continue
        if body[i] == "|":
            cells.append(" ".join("".join(current).split()))
            current = []
            i += 1
            continue
        current.append(body[i])
        i += 1
    cells.append(" ".join("".join(current).split()))
    return tuple(cells)


def is_delimiter(cells: tuple[str, ...]) -> bool:
    """The `|---|---|` line under a table's header."""
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def read_tables(path: pathlib.Path) -> list[Table]:
    """Every table in the file, with the line its header sits on."""
    lines = read_text(path).split("\n")
    source = display(path)
    tables: list[Table] = []
    index = 0
    while index < len(lines):
        if table_cells(lines[index]) is None:
            index += 1
            continue
        start = index
        block: list[tuple[int, tuple[str, ...]]] = []
        while index < len(lines):
            cells = table_cells(lines[index])
            if cells is None:
                break
            block.append((index + 1, cells))
            index += 1
        tables.append(
            Table(
                source,
                start + 1,
                block[0][1],
                tuple(Row(cells, line) for line, cells in block[1:] if not is_delimiter(cells)),
            )
        )
    return tables


def find_table(tables: list[Table], path: pathlib.Path, first: str, count: int) -> Table:
    """The one table whose first column is `first` and whose header holds `count` cells."""
    hits = [table for table in tables if table.columns[:1] == (first,) and len(table.columns) == count]
    if len(hits) != 1:
        seen = "; ".join(
            f"{table.source}:{table.line} {list(table.columns)}"
            for table in tables
            if table.columns[:1] == (first,)
        )
        raise CheckError(
            f"{display(path)}: expected one table whose first column is `{first}` with {count} "
            f"columns, and found {len(hits)}. The tables whose first column is `{first}` are: "
            f"{seen or 'none'}"
        )
    return hits[0]


def column_of(table: Table, prefix: str) -> int:
    """The index of the one column whose label begins with `prefix`."""
    hits = [index for index, label in enumerate(table.columns) if label.startswith(prefix)]
    if len(hits) != 1:
        raise CheckError(
            f"{table.source}:{table.line}: expected one column whose label begins with `{prefix}` "
            f"and found {len(hits)}; the header reads {list(table.columns)}"
        )
    return hits[0]


def row_of(table: Table, prefix: str) -> Row:
    """The one row whose first cell begins with `prefix`."""
    hits = [row for row in table.rows if row.label.startswith(prefix)]
    if len(hits) != 1:
        found = "; ".join(f"{table.source}:{row.line} `{ascii_safe(row.label)}`" for row in hits)
        read = "; ".join(f"`{ascii_safe(row.label)}`" for row in table.rows)
        raise CheckError(
            f"{table.source}:{table.line}: expected one row whose first cell begins with "
            f"`{prefix}` and found {len(hits)}. "
            + (f"The rows that begin with it: {found}" if found else f"The rows read: {read}")
        )
    return hits[0]


def region_row(table: Table, region_column: int, lane: str, region: str) -> Row:
    """The one row of the printed-run table whose lane and region are these."""
    hits = [
        row
        for row in table.rows
        if row.cells[0] == lane
        and region_column < len(row.cells)
        and row.cells[region_column] == region
    ]
    if len(hits) != 1:
        read = "; ".join(
            f"`{row.cells[0]} | {row.cells[region_column]}`"
            for row in table.rows
            if region_column < len(row.cells)
        )
        raise CheckError(
            f"{table.source}:{table.line}: expected one row named `{lane} | {region}` and found "
            f"{len(hits)}. The rows read: {read}"
        )
    return hits[0]


def figures_in(text: str) -> list[tuple[float, str, str]]:
    """Every bound figure in one cell, as (value, unit, as written)."""
    found: list[tuple[float, str, str]] = []
    for match in MULT.finditer(text):
        found.append((float(match.group("num")), "absolute", match.group(0)))
        addend = ADDEND.match(text[match.end() :])
        if addend is not None:
            found.append((float(addend.group("num")), "additive", addend.group(0).strip()))
    for match in ULP.finditer(text):
        found.append((float(match.group("num")), "ulp", match.group(0)))
    if not found:
        bare = BARE.fullmatch(text.strip())
        if bare is not None:
            found.append((float(bare.group("num")), "absolute", bare.group(0)))
    return found


def cell_figures(table: Table, row: Row, index: int) -> list[Figure]:
    """The figures one cell of one row carries, with the cell's own identity."""
    if index >= len(row.cells):
        raise CheckError(
            f"{table.source}:{row.line}: the row holds {len(row.cells)} cells and this check reads "
            f"column {index} of the table at {table.source}:{table.line}"
        )
    cell = row.cells[index]
    where = (
        f"{table.source}:{row.line} "
        f"(`{ascii_safe(row.label)}` / `{ascii_safe(table.columns[index])}`)"
    )
    return [Figure(value, unit, text, where) for value, unit, text in figures_in(cell)]


def row_figures(table: Table, row: Row, columns: tuple[int, ...]) -> list[Figure]:
    """Every figure in a row, over the columns this check reads."""
    found: list[Figure] = []
    for index in columns:
        found.extend(cell_figures(table, row, index))
    return found


def keys(figures: list[Figure]) -> set[tuple[float, str]]:
    """The figures a reading carries, as the pairs a comparison can be made on."""
    return {figure.key for figure in figures}


def pretty(key: tuple[float, str]) -> str:
    """A figure's value and unit, written the way a finding has to read."""
    value, unit = key
    text = f"{value:g}"
    return text if unit == "absolute" else f"{text} ({unit})"


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


def read_gate(path: pathlib.Path) -> dict[str, Figure]:
    """The gate's scalar constants that hold a numeric literal of their own."""
    text = strip_comments(read_text(path))
    source = display(path)
    constants: dict[str, Figure] = {}
    for match in DECLARATION.finditer(text):
        init = match.group("init").strip()
        if NUMBER.fullmatch(init) is None:
            # A derived or referring initialiser - an expression, or another
            # constant's name - is not itself a transcription of a document
            # figure, and it moves with the constant it names.
            continue
        line = text.count("\n", 0, match.start()) + 1
        constants[match.group("name")] = Figure(
            float(init), "absolute", init, f"{source}:{line} ({match.group('name')})"
        )
    if not constants:
        raise CheckError(
            f"{source}: no `constexpr double NAME = <literal>;` declaration was found. This check "
            f"reads the gate's transcribed bounds there, so a revision that states them another "
            f"way has to be told to it rather than read as no constants"
        )
    return constants


def declared_gate(gate: dict[str, Figure], name: str) -> Figure:
    """One named constant of the gate, or an error naming what is missing."""
    if name not in gate:
        raise CheckError(
            f"the gate has no `constexpr double {name} = <literal>;` this reader can read. Either "
            f"the constant was renamed, or it is written as an expression now, and this check has "
            f"to be told rather than read nothing"
        )
    return gate[name]


def read_lanes(
    path: pathlib.Path,
) -> dict[str, tuple[Figure, Figure | None, Figure | None]]:
    """The `BoysLaneContracts()` rows: lane name to its bound, additive and plain term.

    The third is the plain reciprocal's own term, zero on a lane whose forms
    deliver one figure. Where it is not zero the figure the lane publishes for
    that form is the bound plus it, and that sum is what the document rows are
    held to - the row states the term and the document states the figure, which
    is the same fact written the two ways each side writes it.
    """
    text = strip_comments(read_text(path))
    source = display(path)
    match = LANE_ROWS.search(text)
    if match is None:
        raise CheckError(
            f"{source}: no `std::array<LaneContractInfo, N> name = {{{{ ... }}}};` initialiser was "
            f"found. This check reads the library's own lane figures there"
        )
    body = match.group("rows")
    rows = [row for row in split_members(body, ",") if row]
    if len(rows) != int(match.group("count")):
        raise CheckError(
            f"{source}: the initialiser declares {match.group('count')} lane rows and holds "
            f"{len(rows)}"
        )
    body_start = text.count("\n", 0, match.start("rows")) + 1
    lanes: dict[str, tuple[Figure, Figure | None, Figure | None]] = {}
    offset = 0
    for index, row in enumerate(rows):
        at = body.find(row, offset)
        line = body_start + body.count("\n", 0, at)
        offset = at + len(row)
        where = f"{source}:{line} (BoysLaneContracts() row {index + 1})"
        if not (row.startswith("{") and row.endswith("}")):
            raise CheckError(f"{where}: not a braced initialiser ({ascii_safe(row[:60])} ...)")
        members = split_members(row[1:-1], ",")
        if len(members) != LANE_MEMBERS:
            raise CheckError(
                f"{where}: {len(members)} members, and LaneContractInfo has {LANE_MEMBERS} "
                f"(precision, name, bound, additive, source)"
            )
        if re.fullmatch(r'"[^"\\]*"', members[1].strip()) is None:
            raise CheckError(
                f"{where}: the lane name is not a plain string literal ({ascii_safe(members[1])}); "
                f"this check keys its rows by that name"
            )
        lane = members[1].strip()[1:-1]
        bound = members[2].strip()
        additive = members[3].strip()
        plain_additive = members[4].strip()
        for value in (bound, additive, plain_additive):
            if NUMBER.fullmatch(value) is None:
                raise CheckError(
                    f"{where}: `{value}` is not a numeric literal, and this check reads the rows' "
                    f"bound and additive as figures"
                )
        if float(bound) == 0.0:
            raise CheckError(
                f"{where}: the row's bound is zero. A lane with no bound is one this check has to "
                f"be told about rather than read as carrying no figure"
            )
        lanes[lane] = (
            Figure(float(bound), "absolute", bound, f"{where} bound"),
            Figure(float(additive), "additive", additive, f"{where} additive")
            if float(additive) != 0.0
            else None,
            Figure(
                float(bound) + float(plain_additive),
                "absolute",
                f"{bound} + {plain_additive}",
                f"{where} plain-reciprocal figure",
            )
            if float(plain_additive) != 0.0
            else None,
        )
    if not lanes:
        raise CheckError(f"{source}: the initialiser holds rows and no lane")
    return lanes


def one(figures: list[Figure], what: str) -> Figure:
    """The one figure a cell that must carry exactly one figure carries."""
    if len(figures) != 1:
        written = ", ".join(f"{figure.text} ({figure.unit})" for figure in figures) or "none"
        raise CheckError(
            f"{what}: this check reads one bound figure there and found {len(figures)} ({written}). "
            f"A cell that stops carrying exactly one figure is a construct it cannot read"
        )
    return figures[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="the form the other tools take: report and exit non-zero on a difference",
    )
    parser.add_argument("--gate", default=str(GATE), help="the accuracy gate's translation unit")
    parser.add_argument("--header", default=str(HEADER), help="the public header")
    parser.add_argument("--readme", default=str(README), help="the README")
    parser.add_argument("--library", default=str(LIBRARY), help="the lane rows' translation unit")
    args = parser.parse_args()

    gate_path, header_path = pathlib.Path(args.gate), pathlib.Path(args.header)
    readme_path, library_path = pathlib.Path(args.readme), pathlib.Path(args.library)

    findings: list[str] = []
    lines: list[str] = []
    try:
        gate = read_gate(gate_path)
        lanes = read_lanes(library_path)

        header = find_table(read_tables(header_path), header_path, "Lane", 1 + len(REGIONS))
        region_column = {key: column_of(header, label) for key, label in REGIONS.items()}

        readme_tables = read_tables(readme_path)
        contract = find_table(readme_tables, readme_path, "Lane", 2)
        run = find_table(readme_tables, readme_path, "Lane", 4)
        contract_bound = column_of(contract, "Error bound")
        run_region = column_of(run, "Region")
        run_bound = column_of(run, "Bound claimed")

        lines.append("the gate's transcribed constants (the figures this check is about)")
        for tie in TIES:
            figure = declared_gate(gate, tie.constant)
            lines.append(f"  {tie.constant:<22} {figure.text:<9} {figure.where}")
        for constant, primary in ALIASES:
            alias = declared_gate(gate, constant)
            lines.append(
                f"  {constant:<22} {alias.text:<9} {alias.where} (the {primary}'s figure again)"
            )
        for untied in UNTIED:
            figure = declared_gate(gate, untied.constant)
            lines.append(f"  {untied.constant:<22} {figure.text:<9} {figure.where} (not tied)")

        lines.append(f"\nthe library's own lane rows ({display(library_path)}, BoysLaneContracts())")
        for lane, (bound, additive, plain) in lanes.items():
            term = f" + {additive.text}" if additive is not None else ""
            plain_term = f", or {plain.text} in the plain form" if plain is not None else ""
            lines.append(
                f"  {lane:<12} {bound.text}{term:<12}{plain_term:<34} "
                f"{bound.where.rsplit(' ', 1)[0]}"
            )

        for table, name, columns in (
            (header, "the header's contract table", tuple(region_column.values())),
            (contract, "the README's contract table", (contract_bound,)),
            (run, "the README's printed-run table", (run_bound,)),
        ):
            lines.append(
                f"\n{name} ({table.source}:{table.line}), columns: "
                f"{', '.join(ascii_safe(column) for column in table.columns)}"
            )
            for row in table.rows:
                figures = row_figures(table, row, columns)
                written = ", ".join(f"{figure.text} [{figure.unit}]" for figure in figures) or "-"
                lines.append(
                    f"  {table.source}:{row.line}  {ascii_safe(row.label):<44} {written}"
                )

        # --- the comparisons -------------------------------------------------
        #
        # A tie declares the unit its constant is stated in - the native half
        # lane's is ULP of the returned value, not an absolute figure - so the
        # figure compared is the constant's value with the tie's unit, and an
        # absolute cell can never satisfy the ULP tie or the other way round.
        tied: set[tuple[str, int]] = set()
        expected: dict[tuple[str, int], set[tuple[float, str]]] = {}
        for tie in TIES:
            figure = declared_gate(gate, tie.constant)
            if figure.unit != "absolute":
                raise CheckError(
                    f"{figure.where}: read as {figure.unit}, and a tie declares a constant of the "
                    f"gate, whose figures are absolute"
                )
            stated = f"{figure.text} ({tie.unit})" if tie.unit != "absolute" else figure.text
            wanted = (figure.value, tie.unit)
            header_row = row_of(header, tie.header)
            tied.add((header.source, header_row.line))
            for key in tie.columns:
                found = one(
                    cell_figures(header, header_row, region_column[key]),
                    f"{header.source}:{header_row.line} (`{tie.header}` / `{REGIONS[key]}`)",
                )
                if found.key != wanted:
                    findings.append(
                        f"{tie.constant} is {stated} at {figure.where}, and the header's "
                        f"`{tie.header}` row states {found.text} ({found.unit}) in its "
                        f"{REGIONS[key]} cell at {found.where}"
                    )
            contract_row = row_of(contract, tie.readme)
            tied.add((contract.source, contract_row.line))
            expected.setdefault((contract.source, contract_row.line), set()).add(wanted)
            for lane, region in tie.regions:
                run_row = region_row(run, run_region, lane, region)
                tied.add((run.source, run_row.line))
                found = one(
                    cell_figures(run, run_row, run_bound),
                    f"{run.source}:{run_row.line} (`{lane}` / `{region}`)",
                )
                if found.key != wanted:
                    findings.append(
                        f"{tie.constant} is {stated} at {figure.where}, and the README's printed "
                        f"run states {found.text} ({found.unit}) for `{lane} | {region}` at "
                        f"{found.where}"
                    )

        # An alias states the primary's figure, and is held to the primary's
        # cells as well, so neither can drift while the other stays put.
        for constant, primary in ALIASES:
            alias = declared_gate(gate, constant)
            source = declared_gate(gate, primary)
            if alias.value != source.value:
                findings.append(
                    f"{constant} is {alias.text} at {alias.where}, and it states the {primary}'s "
                    f"figure, which is {source.text} at {source.where}"
                )
                continue
            held = False
            for tie in TIES:
                if tie.constant != primary:
                    continue
                held = True
                header_row = row_of(header, tie.header)
                for key in tie.columns:
                    found = one(
                        cell_figures(header, header_row, region_column[key]),
                        f"{header.source}:{header_row.line} (`{tie.header}` / `{REGIONS[key]}`)",
                    )
                    if found.key != alias.key:
                        findings.append(
                            f"{constant} is {alias.text} at {alias.where}, and the header's "
                            f"`{tie.header}` row states {found.text} in its {REGIONS[key]} cell at "
                            f"{found.where}"
                        )
            if not held:
                raise CheckError(
                    f"the alias list holds {constant} against {primary}, and no tie declares "
                    f"{primary}: an alias is held to the cells its primary is tied to, and a tie "
                    f"table that drops one has to drop the alias with it"
                )

        for tie in LIBRARY_TIES:
            if tie.row not in lanes:
                raise CheckError(
                    f"{display(library_path)}: no lane row named `{tie.row}`. The rows read are: "
                    f"{', '.join(sorted(lanes))}"
                )
            bound, additive, plain = lanes[tie.row]
            figures = [bound] + ([additive] if additive is not None else [])
            # The README's contract row is the one cell that states every figure
            # a lane publishes, so it carries the plain form's own figure too.
            # The header's table has a cell per region and states the base there,
            # so only the README row is held to the third figure - holding the
            # header to it would need one figure per cell where it states one.
            readme_figures = figures + ([plain] if plain is not None else [])
            where = bound.where.rsplit(" ", 1)[0]
            contract_row = row_of(contract, tie.readme)
            tied.add((contract.source, contract_row.line))
            # A figure a library row states is tied to the README's row the same
            # way a gate constant's is, so the set sweep below reads both. The
            # plain form's figure is the one figure on this side that no constant
            # of the gate transcribes: the row states the term, the document
            # states the figure, and the sum is the tie.
            for figure in readme_figures:
                expected.setdefault((contract.source, contract_row.line), set()).add(figure.key)
            found = keys(row_figures(contract, contract_row, (contract_bound,)))
            if found != keys(readme_figures):
                findings.append(
                    f"the library's `{tie.row}` row carries "
                    f"{', '.join(pretty(figure.key) for figure in readme_figures)} at {where}, and the "
                    f"README row `{ascii_safe(tie.readme)}` at {contract.source}:"
                    f"{contract_row.line} states "
                    f"{', '.join(sorted(pretty(key) for key in found)) or 'nothing'}"
                )
            if tie.header is not None:
                header_row = row_of(header, tie.header)
                tied.add((header.source, header_row.line))
                found = keys(row_figures(header, header_row, tuple(region_column.values())))
                if found != keys(figures):
                    findings.append(
                        f"the library's `{tie.row}` row carries "
                        f"{', '.join(pretty(figure.key) for figure in figures)} at {where}, and "
                        f"the header's `{tie.header}` row at {header.source}:{header_row.line} "
                        f"states {', '.join(sorted(pretty(key) for key in found)) or 'nothing'}"
                    )

        # The README's contract rows, as sets: a row states every figure its lane
        # publishes, and the figures tied to the row - the gate's constants and
        # the library's own row - are all of them. This runs after both tie
        # tables for that reason.
        for (source, line), wanted in expected.items():
            row = next(row for row in contract.rows if row.line == line)
            found = keys(row_figures(contract, row, (contract_bound,)))
            where = f"{source}:{line} (`{ascii_safe(row.label)}`)"
            missing = wanted - found
            extra = found - wanted
            if missing:
                findings.append(
                    f"the figures tied to {where} state "
                    f"{', '.join(sorted(pretty(key) for key in missing))}, and the row does not: "
                    f"it states {', '.join(sorted(pretty(key) for key in found)) or 'nothing'}"
                )
            if extra:
                findings.append(
                    f"{where} states {', '.join(sorted(pretty(key) for key in extra))}, and "
                    f"neither a constant of the gate nor a row of BoysLaneContracts() tied to it "
                    f"carries it: the row would be publishing a figure nothing in the library "
                    f"states"
                )

        # The withdrawn bound: the header must not claim it outside region A.
        withdrawn = declared_gate(gate, UNTIED[0].constant)
        header_row = row_of(header, TIES[0].header)
        for key in ("band", "B"):
            found = cell_figures(header, header_row, region_column[key])
            if withdrawn.key in keys(found):
                findings.append(
                    f"{UNTIED[0].constant} is {withdrawn.text} at {withdrawn.where}, the bound the "
                    f"gate records as withdrawn, and the header's `{TIES[0].header}` row claims it "
                    f"again in its {REGIONS[key]} cell at {found[0].where}"
                )

        # --- the sweep, and the coverage the declaration lists owe ------------
        declared = {tie.constant for tie in TIES}
        declared |= {constant for constant, _ in ALIASES}
        declared |= {untied.constant for untied in UNTIED}
        documented: dict[tuple[float, str], str] = {}
        for tie in TIES:
            figure = declared_gate(gate, tie.constant)
            documented.setdefault(figure.key, figure.where)
        for name, figure in sorted(gate.items()):
            if (
                name not in declared
                and figure.unit == "absolute"
                # 8 is a literal any file may carry for its own reasons, so the
                # sweep does not claim it; the constant that states it is tied
                # by name like the rest.
                and figure.key[0] != 8.0
                and figure.key in documented
            ):
                findings.append(
                    f"{name} is {figure.text} at {figure.where}, which is the figure "
                    f"{documented[figure.key]} ties to a document, and it is declared in neither "
                    f"the tie table nor the untied list of tools/check_bound_transcripts.py: a "
                    f"second transcription of a documented figure is exactly what this check "
                    f"exists for, so it has to be tied or named"
                )

        coverage = (
            ("header", header, tuple(region_column.values())),
            ("readme-contract", contract, (contract_bound,)),
            ("readme-run", run, (run_bound,)),
        )
        read = 0
        for name, table, columns in coverage:
            for row in table.rows:
                found = row_figures(table, row, columns)
                read += len(found)
                if not found or (table.source, row.line) in tied:
                    continue
                if any(
                    uncovered.table == name and uncovered.label == row.label
                    for uncovered in UNCOVERED
                ):
                    continue
                findings.append(
                    f"{table.source}:{row.line} (`{ascii_safe(row.label)}`) carries "
                    f"{', '.join(pretty(figure.key) for figure in found)} and no tie holds it: a "
                    f"row that states a bound figure has to be tied to a constant or named in the "
                    f"UNCOVERED list of tools/check_bound_transcripts.py with the reason it is not"
                )

        for uncovered in UNCOVERED:
            if uncovered.table == "readme-run":
                row = region_row(run, run_region, uncovered.label, uncovered.region)
                if not row_figures(run, row, (run_bound,)):
                    findings.append(
                        f"{run.source}:{row.line} (`{uncovered.label}` / `{uncovered.region}`) is "
                        f"named as uncovered and carries no figure this check can read"
                    )
            elif uncovered.table == "header":
                row_of(header, uncovered.label)
            else:
                row_of(contract, uncovered.label)

        if read == 0:
            raise CheckError(
                "no bound figure was read from any document: this check has no subject. Either the "
                "tables stopped carrying figures or this reader no longer reads them"
            )
    except CheckError as error:
        print(f"check_bound_transcripts: {ascii_safe(str(error))}", file=sys.stderr)
        return 1

    out = sys.stdout
    print(
        "check_bound_transcripts: the bound figures the accuracy gate transcribes, against the "
        "documents that state them\n"
    )
    for line in lines:
        print(ascii_safe(line), file=out)

    print(f"\nties confirmed: {len(TIES)} constants and {len(LIBRARY_TIES)} library rows", file=out)
    for tie in TIES:
        figure = declared_gate(gate, tie.constant)
        places = [f"the README's `{tie.readme}` (as a set)"]
        places.extend(f"`{lane} | {region}` in the README's printed run" for lane, region in tie.regions)
        print(
            f"  {tie.constant:<22} {pretty(figure.key):<14} = the header's `{tie.header}` "
            f"({', '.join(tie.columns)}) = " + " = ".join(places),
            file=out,
        )
    for constant, primary in ALIASES:
        print(
            f"  {constant:<22} {pretty(declared_gate(gate, constant).key):<14} = the same figure "
            f"as {primary}, held to the same cells",
            file=out,
        )
    for tie in LIBRARY_TIES:
        bound, additive, plain = lanes[tie.row]
        figures = [bound] + ([additive] if additive is not None else [])
        readme_figures = figures + ([plain] if plain is not None else [])
        print(
            f"  BoysLaneContracts() {tie.row:<12} "
            f"{', '.join(pretty(figure.key) for figure in readme_figures):<22} = the README's "
            f"`{tie.readme}` (as a set)"
            + (f" = the header's `{tie.header}` (as a set)" if tie.header else ""),
            file=out,
        )

    print("\nnot checked, and why", file=out)
    for untied in UNTIED:
        figure = declared_gate(gate, untied.constant)
        print(f"  {untied.constant} = {figure.text} ({figure.where}): {untied.reason}", file=out)
    for uncovered in UNCOVERED:
        row = region_row(run, run_region, uncovered.label, uncovered.region)
        figures = ", ".join(pretty(figure.key) for figure in row_figures(run, row, (run_bound,)))
        print(
            f"  {run.source}:{row.line} (`{uncovered.label}` / `{uncovered.region}`, "
            f"{figures or 'no figure'}): {uncovered.reason}",
            file=out,
        )

    if findings:
        print(f"\n{len(findings)} finding(s):", file=out)
        for finding in findings:
            print(f"\nMISMATCH: {ascii_safe(finding)}", file=out)
        print(
            "\nA figure a document states and the gate transcribes is one fact with two copies of "
            "it. Changing one of them without the other leaves the gate certifying its own "
            "transcription, and every run of it green. Make the two agree, or - where they are "
            "genuinely no longer the same fact - say so in the tie table of "
            "tools/check_bound_transcripts.py.",
            file=out,
        )
        return 1

    print(
        f"\nverdict: clean - {len(TIES)} transcribed constants and {len(LIBRARY_TIES)} library rows "
        f"agree with the figures README.md and include/boys/boys.hpp state, over the ties above. "
        f"{read} figures were read from the documents' tables, {len(gate)} numeric-literal "
        f"constants from the gate and {len(lanes)} lane rows from the library",
        file=out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
