#!/usr/bin/env python3
"""Put an emitted seam's rows into the committed seam, and prove they arrived whole.

`boys-option-probe --emit-defaults` and `boys-device-probe --emit-defaults` write a
header carrying the rows a run measured. The committed seam carries the rows a build
resolves to. This copies the rows of one into the other.

It copies **rows and nothing else**. The prose around them is written by a person and
is the part a splice gets wrong: the last one replaced the rows and left the previous
run's sentence standing, so the file named a run it had not come from and nothing
noticed until the emitted file was read beside it. A tool that rewrites the whole file
would have made the same mistake faster, so this one refuses to touch a line that is
not a row.

**A row the emission states a marker for is a row the emission places, and it is
written. A row it states none for is a class the emission did not place, and the seam
keeps its own row for that class.** The marker is the emission's own statement of how
a row was reached, and a run states one only for a class it measured: the rows an
emission reaches no marker for are the table in force's own rows, which it writes back
because a replacement is read instead of that file, and the fallback combinations a
file states for the classes it ranked no cell of. Either way a row the emission did not
place is a row it has no measurement of, so the seam's own row is the one that stands,
cells and marker together.

Rows are matched by class and never by position. A class is the three cells
`X(device, precision, shape, ...)`, and an emission lists the table in force's rows
before its own, so the two files run in different orders and a splice by position
writes one class's row under another class's key.

What it checks, and refuses on:

* every row the emission places is written, and the row that arrives is byte-identical
  to the emission's, read back after the write rather than assumed from the write;
* a row the emission states no marker for is a class the seam carries, so that nothing
  the emission states is dropped in silence;
* no class is carried twice by either file, because one class is one row;
* every run a row names is a run the seam's prose names. The prose names each run by its
  day and the record cell states it as `<machine>-<date>`, so the day is the token the two
  are read through, and a date is read wherever it appears rather than a sentence parsed.
  A row whose record names a day the prose does not is the sentence the splice would leave
  standing over rows it does not describe; a record naming no day is a run the prose cannot
  be read against at all.

What it does not check, and why, of the sentences a splice can outlive:

* the run's hour and its protocol - the processors, the tier present, the seed, the passes,
  the rounds, the arguments, the order. No row states one: a record is `<machine>-<date>`,
  and a check of these would compare the sentence against itself.
* the machine as the prose spells it, beside the machine word a record carries, `Quadro T1000`
  beside `quadro-t1000`. Both are readable, and a rule that equated the two spellings would be
  this tool's invention rather than a reading of the file.
* a sentence about what the rows are: that they are measurements taken on a machine, that a
  half's rows carry one record name. A splice can falsify either, and finding the sentence
  means reading English, where the wording is a person's to change: a check anchored on a
  wording disarms in silence when the wording moves. The cell behind the first is held by
  `check_default_capturability.py`, which refuses a measurement that names no run; the second
  is a sentence a partial splice obliges a person to rewrite, and the tool cannot read the
  rewrite back.
* the count of cells a row carries. It is readable and it is not refused on: an emission
  written before the two provenance cells existed carries rows of another count, and that
  emission is one this tool is asked to splice. The row macro is what refuses a row that does
  not expand.

Usage:

    python tools/splice_default_rows.py --emitted <file> --seam <file> [--device-only]
    python tools/splice_default_rows.py --emitted <file> --seam <file> --check
    python tools/splice_default_rows.py --control

Exit status: 0 when the seam carries the emitted rows; 1 when it does not and was not
asked to change; 2 when the two files cannot be reconciled - a placed row the seam has
no class for, a row the emission states no marker for and the seam has no class to keep
for it, a class one of them carries twice, or a row whose run the seam's prose does not
name.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import tempfile
from pathlib import Path

# A row and the marker comment over it. A row is `X(...)` and nothing else; the marker
# is the `/* ... */` run immediately above it, one complete comment per line, which is
# where a run states how the row was reached. They are read as a pair because a row the
# emission states no marker for is a row it did not place.
MARKER = re.compile(r"^\s*/\*.*\*/\s*$")

DEVICE = "X(kDevice"

# Every line of the rows block ends with a backslash, because the whole list is one
# macro definition: the continuation is the macro's, not the row's, so it says nothing
# about where a row begins or ends. What does is the row's own shape - it opens with
# `X(` and closes with the `)` that balances it - read on each line's content.
CONTINUED = "\\"
ROW_OPENS = re.compile(r"^\s*X\(")
ROW_CLOSES = re.compile(r"\)\s*$")

# The cells a row is keyed on: the device, the precision lane and the shape. Two rows
# with the same three are two answers for one class, which is not a table.
KEY_CELLS = 3

# The two provenance cells a row ends with, and the day a record names. A row that states
# neither has the cells of a seam written before they existed, and is read as naming no run.
BASIS_CELLS = ("RowBasis::kMeasured", "RowBasis::kChosen")
DAY = re.compile(r"\d{4}-\d{2}-\d{2}")


class Refusal(Exception):
    """The two files cannot be reconciled; the message names what is wrong with them."""


def content(line: str) -> str:
    """A physical line without the macro's continuation backslash."""
    stripped = line.rstrip()
    if stripped.endswith(CONTINUED):
        stripped = stripped[:-1]
    return stripped.rstrip()


def elements(text: str) -> list[tuple[str, list[str], list[str]]]:
    """The text as elements: a row with its marker, or lines that are neither.

    Each element is `(kind, marker_lines, body_lines)`. A marker is the run of
    `/* ... */` lines immediately above a row; a line that is neither a marker nor
    part of a row ends the run, so a paragraph of prose above a block is not carried
    into the next row's marker.
    """
    lines = text.splitlines()
    found: list[tuple[str, list[str], list[str]]] = []
    pending: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]

        if ROW_OPENS.match(content(line)):
            body = [line.rstrip()]
            index += 1
            while not ROW_CLOSES.search(content(body[-1])) and index < len(lines):
                body.append(lines[index].rstrip())
                index += 1
            found.append(("row", pending[-12:], body))
            pending = []
            continue

        if MARKER.match(content(line)):
            pending.append(line.rstrip())
            index += 1
            continue

        if pending:
            found.append(("other", [], pending))
            pending = []

        found.append(("other", [], [line.rstrip()]))
        index += 1

    if pending:
        found.append(("other", [], pending))

    return found


def read_rows(text: str) -> list[tuple[list[str], list[str]]]:
    return [(marker, body) for kind, marker, body in elements(text) if kind == "row"]


def is_device(row: tuple[list[str], list[str]]) -> bool:
    return any(DEVICE in line for line in row[1])


def cells(row: tuple[list[str], list[str]]) -> list[str]:
    """A row's cells, in the order the row macro writes them."""
    text = " ".join(content(line) for line in row[1])
    inside = text[text.index("(") + 1 : text.rindex(")")]

    return [cell.strip() for cell in inside.split(",")]


def class_of(row: tuple[list[str], list[str]]) -> str:
    """The class a row is keyed on, spelled as the seam spells its three cells."""
    return " ".join(cells(row)[:KEY_CELLS])


def provenance(row: tuple[list[str], list[str]]) -> tuple[str, str] | None:
    """The (basis, record) a row ends with, or None when it states no provenance."""
    stated = cells(row)

    return (stated[-2], stated[-1]) if len(stated) >= 2 and stated[-2] in BASIS_CELLS else None


def prose_of(text: str) -> str:
    """The seam's own prose: every line this tool does not write.

    A marker is the emission's statement about its row and is written with it, so a marker is not
    read here: a marker naming a run would otherwise corroborate the row carrying it.
    """
    return "\n".join(line for kind, _marker, body in elements(text) if kind != "row" for line in body)


def disagreement(spliced: str) -> str | None:
    """The row whose run the file's prose does not name, or None when the two agree.

    A row names a run and the prose beside the rows names runs the same way, by their day, so a
    day is what the two are read through - the cells and the prose are compared as tokens and not
    as sentences.
    """
    named = set(DAY.findall(prose_of(spliced)))

    for marker, body in read_rows(spliced):
        row = (marker, body)
        stated = provenance(row)

        if stated is None:
            continue

        record = stated[1].strip('"')

        if not record:
            continue

        day = DAY.search(record)

        if day is None:
            return (f"the row for the class {class_of(row)} names the run {record}, which states no "
                    "day, and the prose beside the rows names each run by its day")

        if day.group() not in named:
            return (f"the row for the class {class_of(row)} names the run {record}, and the prose "
                    f"beside the rows names no run of {day.group()}: the rows and the sentences "
                    "around them would be left disagreeing")

    return None


def splice(
    seam_text: str, placed: dict[str, tuple[list[str], list[str]]], device_only: bool
) -> tuple[str, set[str], set[str]]:
    """The seam with the rows the emission places written over its own, class for class.

    Returns the text, the classes written, and the classes the seam carried in scope.
    """
    out: list[str] = []
    written: set[str] = set()
    carried: set[str] = set()

    for kind, marker, body in elements(seam_text):
        if kind != "row":
            out.extend(marker)
            out.extend(body)
            continue

        row = ("", body)

        if device_only and not is_device(row):
            out.extend(marker)
            out.extend(body)
            continue

        klass = class_of(row)

        if klass in carried:
            raise Refusal(f"the seam carries the class {klass} twice, and one class is one row")

        carried.add(klass)
        replacement = placed.get(klass)

        if replacement is None:
            out.extend(marker)
            out.extend(body)
            continue

        out.extend(replacement[0])
        out.extend(replacement[1])
        written.add(klass)

    return "\n".join(out) + "\n", written, carried


def silent(argv: list[str]) -> tuple[int, str, str]:
    """`main` over one argument list, with the streams it writes to captured."""
    out, err = io.StringIO(), io.StringIO()
    kept = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err

    try:
        code = main(argv)
    finally:
        sys.stdout, sys.stderr = kept

    return code, out.getvalue(), err.getvalue()


# The control's fixtures. The pair is small on purpose: a seam of two classes, each row with the
# marker the table in force carries, and an emission whose first row places the second class and
# whose second row states the first class with no marker. The emission lists its two rows in the
# seam's other order, and states different cells for both classes, so a splice that matched rows by
# position, or that wrote the emission's row where the emission placed none, produces a text that
# is not the one below.
CONTROL_SEAM = (
    "#pragma once\n"
    "\n"
    "/// A two-row seam for the control: the table in force, both rows with the marker it carries.\n"
    "\n"
    "#define BOYS_BUILD_DEFAULT_ROWS(X)\\\n"
    "    /* the table in force took this row at 1.00 ns per argument */\\\n"
    "    X(kHost, kFp64, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\\\n"
    "      PackAxis::kArguments, FitGranularity::kUniform, DivisionForm::kPlainReciprocal,\\\n"
    "      RegionBExp::kAccurate)\\\n"
    "    /* the table in force took this row at 2.00 ns per argument */\\\n"
    "    X(kHost, kFp32, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\\\n"
    "      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kPlainReciprocal,\\\n"
    "      RegionBExp::kFast)\\\n"
)

CONTROL_EMISSION = (
    "#pragma once\n"
    "\n"
    "/// A two-row emission for the control: the class it placed, then one it states no marker\n"
    "/// for.\n"
    "\n"
    "#define BOYS_BUILD_DEFAULT_ROWS(X)\\\n"
    "    /* measured: 0.50 ns per argument on this host, reached by ordered */\\\n"
    "    X(kHost, kFp32, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\\\n"
    "      PackAxis::kOrders, FitGranularity::kUniform, DivisionForm::kRefinedReciprocal,\\\n"
    "      RegionBExp::kAccurate)\\\n"
    "    X(kHost, kFp64, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\\\n"
    "      PackAxis::kArguments, FitGranularity::kCoarsest, DivisionForm::kRefinedReciprocal,\\\n"
    "      RegionBExp::kFast)\\\n"
)

# What the pair implies: the seam's own order, the seam's own row and marker for the class the
# emission placed none for, and the emission's row and marker for the class it placed.
CONTROL_SPLICED = (
    "#pragma once\n"
    "\n"
    "/// A two-row seam for the control: the table in force, both rows with the marker it carries.\n"
    "\n"
    "#define BOYS_BUILD_DEFAULT_ROWS(X)\\\n"
    "    /* the table in force took this row at 1.00 ns per argument */\\\n"
    "    X(kHost, kFp64, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\\\n"
    "      PackAxis::kArguments, FitGranularity::kUniform, DivisionForm::kPlainReciprocal,\\\n"
    "      RegionBExp::kAccurate)\\\n"
    "    /* measured: 0.50 ns per argument on this host, reached by ordered */\\\n"
    "    X(kHost, kFp32, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\\\n"
    "      PackAxis::kOrders, FitGranularity::kUniform, DivisionForm::kRefinedReciprocal,\\\n"
    "      RegionBExp::kAccurate)\\\n"
)

# The fixture's two rows, renamed to a class the seam carries no row of: the planted defect of each
# half - a placed row the seam has no class for, and a markerless one it has no row to keep.
CONTROL_PLACED_ROW = "X(kHost, kFp32, kSingle,"
CONTROL_MARKERLESS_ROW = "X(kHost, kFp64, kSingle,"
CONTROL_ABSENT_CLASS = "X(kHost, kFp64, kFixedN,"

# One class carried twice: the first class's row written a second time under the same three cells,
# which is two rows for a class the emission states one row for.
CONTROL_TWICE = CONTROL_SPLICED + (
    "    X(kHost, kFp64, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\\\n"
    "      PackAxis::kArguments, FitGranularity::kUniform, DivisionForm::kPlainReciprocal,\\\n"
    "      RegionBExp::kAccurate)\\\n"
)

# The run cells and the sentence beside them. One row is written three times: as the seam whose
# prose names the run its row came from, as the earlier run's row, and as the row the next run
# emits, which differs from the seam's own in the figure its marker states. Two splices follow: into
# the seam whose prose names the arriving run's day, which is the write the rows and the prose agree
# on, and into the earlier seam, which leaves a sentence naming the run of the row that was
# replaced - the disagreement this pair is here to refuse.
CONTROL_RUN_SEAM = (
    "#pragma once\n"
    "\n"
    "/// A one-row seam for the control whose prose names the run its row came from, 2026-01-09.\n"
    "\n"
    "#define BOYS_BUILD_DEFAULT_ROWS(X)\\\n"
    "    /* measured: 1.00 ns per argument on this host, reached by vote */\\\n"
    "    X(kHost, kFp64, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\\\n"
    "      PackAxis::kArguments, FitGranularity::kUniform, DivisionForm::kPlainReciprocal,\\\n"
    "      RegionBExp::kAccurate, RowBasis::kMeasured, \"host-2026-01-09\")\\\n"
)

CONTROL_RUN_EARLIER = CONTROL_RUN_SEAM.replace("2026-01-09", "2026-01-02")
CONTROL_RUN_EMISSION = CONTROL_RUN_SEAM.replace("1.00 ns", "2.00 ns")
CONTROL_RUN_UNDATED = CONTROL_RUN_EMISSION.replace('"host-2026-01-09"', '"host-undated"')

# Two runs in one half, and a splice over the first of them. The prose names both days and the
# emission names the first run's day, so both rows stand and the write is taken: what is refused is
# the row that arrived and not the half it arrived in. An emission writes back the table in force's
# row, record and all, for a class its run did not measure, so a half holding two runs is a file the
# emitters produce.
CONTROL_RUNS_SEAM = (
    "#pragma once\n"
    "\n"
    "/// A two-row seam for the control: the runs 2026-01-02 and 2026-01-16, each named here.\n"
    "\n"
    "#define BOYS_BUILD_DEFAULT_ROWS(X)\\\n"
    "    /* measured: 0.50 ns per argument on this host, reached by ordered */\\\n"
    "    X(kHost, kFp64, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\\\n"
    "      PackAxis::kArguments, FitGranularity::kUniform, DivisionForm::kPlainReciprocal,\\\n"
    "      RegionBExp::kAccurate, RowBasis::kMeasured, \"host-2026-01-02\")\\\n"
    "    /* measured: 3.00 ns per argument on this host, reached by vote */\\\n"
    "    X(kHost, kFp32, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\\\n"
    "      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kPlainReciprocal,\\\n"
    "      RegionBExp::kFast, RowBasis::kMeasured, \"host-2026-01-16\")\\\n"
)

# What that splice implies: the first row replaced by the emission's own, marker and cells, and the
# second left as the seam carries it.
CONTROL_RUNS_SPLICED = CONTROL_RUNS_SEAM.replace(
    "/* measured: 0.50 ns per argument on this host, reached by ordered */",
    "/* measured: 1.00 ns per argument on this host, reached by vote */",
)


def control() -> int:
    """Splice the fixture pair, and require each planted defect refused."""
    problems: list[str] = []
    checked = 0

    def recorded(name: str, code: int, required: int) -> None:
        nonlocal checked
        checked += 1
        verdict = "caught" if code == required else "NOT caught"
        print(f"control: {name}: exit {code}, {required} required - {verdict}")

        if code != required:
            problems.append(f"{name}: exit {code} where {required} was required")

    with tempfile.TemporaryDirectory(prefix="splice-control-") as home:
        work = Path(home)
        seam = work / "seam.hpp"
        emission = work / "emission.hpp"
        seam.write_text(CONTROL_SEAM, encoding="utf-8", newline="\n")
        emission.write_text(CONTROL_EMISSION, encoding="utf-8", newline="\n")

        pair = ["--emitted", str(emission), "--seam", str(seam)]

        code, _out, _err = silent([*pair, "--check"])
        recorded("the seam before the splice, --check", code, 1)

        code, out, err = silent(pair)
        recorded("the splice of the fixture pair", code, 0)
        print(f"control: it printed {out.strip() or err.strip() or '(nothing)'}")

        arrived = seam.read_bytes()

        if arrived == CONTROL_SPLICED.encode("utf-8"):
            print("control: the seam the splice left is the seam the pair implies, byte for byte")
        else:
            problems.append("the seam the splice left is not the seam the pair implies")
            print("control: the seam the splice left is NOT the seam the pair implies")

        code, _out, _err = silent([*pair, "--check"])
        recorded("the spliced seam, --check", code, 0)

        # A seam whose lines end the other way is spliced the same and keeps its own ending: the
        # rows are what this writes, and a file whose every line changed ending would be an edit of
        # the whole file that no row of the emission asks for.
        crooked = work / "crooked.hpp"
        crooked.write_bytes(CONTROL_SEAM.replace("\n", "\r\n").encode("utf-8"))
        code, _out, _err = silent(["--emitted", str(emission), "--seam", str(crooked)])
        recorded("a seam whose lines end the other way", code, 0)

        if crooked.read_bytes() != CONTROL_SPLICED.replace("\n", "\r\n").encode("utf-8"):
            problems.append("the splice changed the line ending of the seam it wrote")

        # A row the emission places whose class the seam carries none of: writing it would name a
        # class the seam has no place for, and keeping the seam's is not possible.
        planted = work / "planted-emission.hpp"
        planted.write_text(CONTROL_EMISSION.replace(CONTROL_PLACED_ROW, CONTROL_ABSENT_CLASS),
                           encoding="utf-8", newline="\n")
        code, _out, err = silent(["--emitted", str(planted), *pair[2:]])
        recorded("a placed row the seam has no class for", code, 2)
        print(f"control: it printed {err.strip() or '(nothing on the error stream)'}")

        if seam.read_bytes() != arrived:
            problems.append("the refused splice wrote to the seam")

        # A row the emission states no marker for whose class the seam carries none of either: the
        # emission did not place it and the seam has no row of its own to keep, so nothing of it
        # can be written and nothing may be dropped in silence.
        planted.write_text(CONTROL_EMISSION.replace(CONTROL_MARKERLESS_ROW, CONTROL_ABSENT_CLASS),
                           encoding="utf-8", newline="\n")
        code, _out, err = silent(["--emitted", str(planted), *pair[2:]])
        recorded("a row with no marker whose class the seam has none of", code, 2)
        print(f"control: it printed {err.strip() or '(nothing on the error stream)'}")

        if seam.read_bytes() != arrived:
            problems.append("the refused splice wrote to the seam")

        # One class carried twice, which is two rows for one class and not a table.
        twice = work / "twice.hpp"
        twice.write_text(CONTROL_TWICE, encoding="utf-8", newline="\n")
        code, _out, err = silent(["--emitted", str(emission), "--seam", str(twice)])
        recorded("a class the seam carries twice", code, 2)
        print(f"control: it printed {err.strip() or '(nothing on the error stream)'}")

        run_emission = work / "run-emission.hpp"
        run_emission.write_text(CONTROL_RUN_EMISSION, encoding="utf-8", newline="\n")
        current = work / "current.hpp"
        current.write_text(CONTROL_RUN_SEAM, encoding="utf-8", newline="\n")
        run_pair = ["--emitted", str(run_emission), "--seam", str(current)]

        # The run the row names is a run the prose names too, so this is the write the two agree on.
        code, _out, _err = silent(run_pair)
        recorded("a seam whose prose names the run the arriving row names", code, 0)

        if current.read_bytes() != CONTROL_RUN_EMISSION.encode("utf-8"):
            problems.append("the run splice is not the seam its pair implies")

        code, _out, _err = silent([*run_pair, "--check"])
        recorded("the spliced run seam, --check", code, 0)

        # The spliced seam's sentence names the run of the row that was replaced and not the run the
        # row names: the refusal this tool exists to make, planted as the day the prose states.
        stale = work / "stale.hpp"
        stale.write_text(CONTROL_RUN_EARLIER, encoding="utf-8", newline="\n")
        code, _out, err = silent(["--emitted", str(run_emission), "--seam", str(stale)])
        recorded("a seam whose prose names the run of the replaced row", code, 2)
        print(f"control: it printed {err.strip() or '(nothing on the error stream)'}")

        if stale.read_bytes() != CONTROL_RUN_EARLIER.encode("utf-8"):
            problems.append("the refused splice wrote to the seam")

        code, _out, _err = silent(["--emitted", str(run_emission), "--seam", str(stale), "--check"])
        recorded("the stale seam, --check", code, 2)

        # A record that names no day: the run behind the row cannot be read against the prose.
        undated = work / "undated.hpp"
        undated.write_text(CONTROL_RUN_UNDATED, encoding="utf-8", newline="\n")
        code, _out, err = silent(["--emitted", str(undated), "--seam", str(stale)])
        recorded("a row whose record names no day", code, 2)
        print(f"control: it printed {err.strip() or '(nothing on the error stream)'}")

        if stale.read_bytes() != CONTROL_RUN_EARLIER.encode("utf-8"):
            problems.append("the refused splice wrote to the seam")

        # Two runs in one half, each named by the prose: taken, because an emission writes back the
        # table in force's row for a class its run did not measure.
        runs = work / "runs.hpp"
        runs.write_text(CONTROL_RUNS_SEAM, encoding="utf-8", newline="\n")
        earlier = work / "earlier-emission.hpp"
        earlier.write_text(CONTROL_RUN_EARLIER, encoding="utf-8", newline="\n")
        code, _out, _err = silent(["--emitted", str(earlier), "--seam", str(runs)])
        recorded("a half whose rows name two runs, both named by the prose", code, 0)

        if runs.read_bytes() != CONTROL_RUNS_SPLICED.encode("utf-8"):
            problems.append("the splice over the first of two runs is not the seam its pair implies")

    if problems:
        print("splice_default_rows: the control failed: " + "; ".join(problems), file=sys.stderr)
        return 1

    print(f"splice_default_rows: control passed every one of its {checked} check(s) over its own "
          f"fixtures")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--emitted", help="the file a run emitted")
    parser.add_argument("--seam", help="the committed seam to write")
    parser.add_argument("--device-only", action="store_true",
                        help="replace the kDevice rows only, leaving the host rows alone")
    parser.add_argument("--check", action="store_true",
                        help="report whether they agree; change nothing")
    parser.add_argument("--control", action="store_true",
                        help="splice the control's own fixture pair, and exit non-zero unless "
                             "every defect planted in it is caught")
    options = parser.parse_args(argv)

    if options.control:
        return control()

    if not options.emitted or not options.seam:
        print("splice_default_rows: --emitted names the file a run wrote and --seam names the "
              "seam to write; a splice needs both", file=sys.stderr)
        return 2

    emitted_path = Path(options.emitted)
    seam_path = Path(options.seam)
    emitted_text = emitted_path.read_text(encoding="utf-8")
    raw = seam_path.read_bytes()

    # The ending the seam already writes its lines with is the one it keeps: a row is what this
    # writes, and a file whose every line changed ending would be a whole-file edit.
    ending = "\r\n" if b"\r\n" in raw else "\n"
    seam_text = raw.decode("utf-8").replace("\r\n", "\n")

    emitted = read_rows(emitted_text)
    replacement = [row for row in emitted if is_device(row)] if options.device_only else emitted

    if not replacement:
        print(f"splice_default_rows: {emitted_path} states no row to splice", file=sys.stderr)
        return 2

    # What the emission places is what carries a marker; what it does not is what the seam keeps.
    placed: dict[str, tuple[list[str], list[str]]] = {}
    stated: list[str] = []

    for marker, body in replacement:
        klass = class_of((marker, body))

        if klass in placed or klass in stated:
            print(f"splice_default_rows: the emission states the class {klass} twice, and one "
                  "class is one row", file=sys.stderr)
            return 2

        if marker:
            placed[klass] = (marker, body)
        else:
            stated.append(klass)

    try:
        spliced, written, carried = splice(seam_text, placed, options.device_only)
    except Refusal as refusal:
        print(f"splice_default_rows: {refusal}", file=sys.stderr)
        return 2

    for klass in placed:
        if klass not in written:
            print(f"splice_default_rows: the seam carries no row for the class {klass} the "
                  "emission places one for; a class the seam does not carry is a class it "
                  "resolves another way", file=sys.stderr)
            return 2

    for klass in stated:
        if klass not in carried:
            print(f"splice_default_rows: the emission states a row for the class {klass} with no "
                  "marker, and the seam carries no row of that class to keep; nothing of the "
                  "row can be written and nothing of it may be dropped in silence",
                  file=sys.stderr)
            return 2

    # Read back what would be written and compare it against the emission, rather than
    # trusting the construction above.
    arrived = read_rows(spliced)
    arrived_rows = [row for row in arrived if is_device(row)] if options.device_only else arrived
    landed = {class_of(row): row for row in arrived_rows if class_of(row) in placed}

    if landed != placed:
        print("splice_default_rows: the rows that arrive are not the rows the emission "
              "states - refusing to write", file=sys.stderr)
        return 2

    # The rows are not the whole file: the sentences beside them name the same runs, and a splice
    # that moves the rows past those sentences leaves a file that disagrees with itself.
    disagreed = disagreement(spliced)

    if disagreed is not None:
        print(f"splice_default_rows: {disagreed}", file=sys.stderr)
        return 2

    if options.check:
        if spliced == seam_text:
            print(f"splice_default_rows: {seam_path} carries the emission's {len(placed)} placed "
                  f"row(s) already")
            return 0

        print(f"splice_default_rows: {seam_path} does NOT carry the emission's {len(placed)} "
              "placed row(s)", file=sys.stderr)
        return 1

    # newline="": the ending the seam's own lines carry is already in the text, and the platform's
    # separator would be added to it rather than written in its place.
    seam_path.write_text(spliced.replace("\n", ending), encoding="utf-8", newline="")
    print(f"splice_default_rows: {len(placed)} row(s) of {emitted_path.name} written into "
          f"{seam_path}, and the emission's {len(stated)} row(s) with no marker left as the seam "
          f"carries them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
