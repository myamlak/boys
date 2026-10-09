#!/usr/bin/env python3
"""Holds the measured table in `docs/getting-started.md` to the option probe's recorded report.

The document's "the double lane costs" section prints one row per double-lane class: the entry the
class's run ranked first, the figure that entry took, and how many options of the class the run
could not place behind it. Each of those three cells is a figure from
`tests/data/boys_option_probe_report.txt`, and a figure copied out of a run is a transcription
rather than a measurement: it is right when the copy is right and wrong when the copy is wrong,
and only a reading of both can tell which one it is.

This check is that reading. It takes the five cells of a class out of the report - the fastest
entry, its figure in nanoseconds per argument, and the count the report states under it - and
holds the document's rows to them, cell for cell. It also counts the rows: the document's sentence
is one row per question shape for the double lane, so a class the report carries and the table
does not is a hole in the table rather than a class left out on purpose.

What the check does not read, and why:
  * it does not judge the report. Whether the run that report records is this tree's own is
    `tools/check_probe_run.py`'s claim, and the two fail apart so that a stale record and a
    mismatched transcription are not reported as one thing;
  * it does not read the prose around the table. A figure quoted into a sentence is a quotation
    like any other and this reading does not reach it, because a sentence is not a cell a
    comparison can be made of.

Usage:
    python3 tools/check_probe_table.py
    python3 tools/check_probe_table.py --doc build-probe-table/doc.md --report some/report.txt
    python3 tools/check_probe_table.py --control

Exit status is 0 when every cell of the table agrees with the report and every double-lane class
the report carries has a row, and 1 when one does not, when the table or the report cannot be
read, when the document carries no row this check can place, or when a row names an entry the
tree's own host-surface table does not place in a class. A run with `--control` exits 0 when every
planted defect is caught and 1 when one is not.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "getting-started.md"
REPORT = REPO / "tests" / "data" / "boys_option_probe_report.txt"

# The tree's own statement of the host entry surface as shape classes. The entry a report's class
# is asked through is read from it rather than written here: the table has two renderings in the
# tree already, and a third is one more thing that can disagree with the other two.
HOST_SHAPES = REPO / "tools" / "combination_matrix.py"

# The lane the table is about. The document's sentence says so - "one row per question shape, for
# the double lane" - and a report carries four lanes' worth of classes, so a reading that took
# every class for a row would be reading a claim the document does not make.
LANE = "fp64"

# The class line of a report, as the probe writes it: the class, the entries its run measured, the
# entry that run ranked first and the figure that entry took. The rule is the one the table was
# extracted by, which reproduces the committed table cell for cell.
CLASS_LINE = re.compile(
    r"^  (?P<precision>fp64|fp32|fp16|bf16) (?P<shape>[a-z-]+)\s+possible .*?fastest "
    r"(?P<label>\S+) at (?P<ns>[0-9.]+) ns/argument",
    re.M,
)

# The sentence a class's block states under that line. It is searched for between one class line
# and the next rather than within a window of lines: the combination the report prints above it
# wraps to as many lines as the names in it need, and a fixed window is a length nothing fixes.
UNPLACED = re.compile(r"^\s+(?P<count>\d+) could not be placed behind it\b", re.M)

# A row of a markdown table, as four cells between two pipes. The cells are read as text for the
# reason the comparison is of text: the row is a transcription, and a cell that says something
# else entirely is the defect as much as a figure that moved.
ROW = re.compile(r"^\|(?P<question>[^|]*)\|(?P<label>[^|]*)\|(?P<ns>[^|]*)\|(?P<count>[^|]*)\|"
                 r"[ \t]*$", re.M)

# The entry a row is about, named in its first cell. The last such name is the entry: a row's
# question cell is prose that may name another entry in passing, and the cell's own subject is
# written where the cell ends.
ENTRY = re.compile(r"`(?P<entry>Boys[A-Za-z0-9_]*)`")

# The cells of a row, counted from the leading pipe of a markdown table line.
QUESTION, LABEL, NS, COUNT = 1, 2, 3, 4

WORK = REPO / "build-probe-table"


def display(path: pathlib.Path) -> str:
    """A path as this tool names it: relative to the repository when it is inside it."""
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def host_surface() -> dict[str, tuple[str, str]]:
    """Every host entry the tree's own table places, as entry -> (precision, shape)."""
    spec = importlib.util.spec_from_file_location("_combination_matrix", HOST_SHAPES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return {family + module.SUFFIX[precision]: (precision, shape)
            for shape, family in module.HOST_ENTRY.items()
            for precision in module.SUFFIX}


def classes_of(report: str) -> dict[tuple[str, str], dict]:
    """The report's classes, as (precision, shape) -> the three cells a row quotes.

    `cell` is None where the block states no count, which is a report this reading cannot place a
    row against rather than a count of zero.
    """
    lines = report.splitlines()
    found: dict[tuple[str, str], dict] = {}
    starts = [index for index, line in enumerate(lines) if CLASS_LINE.match(line)]

    for position, index in enumerate(starts):
        header = CLASS_LINE.match(lines[index])
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        block = "\n".join(lines[index + 1:end])
        count = UNPLACED.search(block)

        found[(header.group("precision"), header.group("shape"))] = {
            "label": header.group("label"),
            "ns": header.group("ns"),
            "count": count.group("count") if count else None,
        }

    return found


def same_number(one: str, other: str) -> bool:
    """Whether two cells state one figure: compared as numbers where both are, as text otherwise."""
    try:
        return float(one) == float(other)
    except ValueError:
        return one.strip() == other.strip()


def read(doc_path: pathlib.Path, report_path: pathlib.Path) -> tuple[list[str], list[str]]:
    """The document's table against the report: the lines a reader gets, and what is wrong.

    The lines are printed by the caller whatever the verdict. The control below reads its
    fixtures through this same function, so a control that passes is a control over the reading
    and not over a second copy of it.
    """
    lines = ["the measured table in the getting-started document, held to the probe's report"]
    findings: list[str] = []

    for label, path in (("document", doc_path), ("report", report_path)):
        if not path.is_file():
            lines.append(f"  {display(path)}: not a file")
            findings.append(
                f"there is no {label} to read: {display(path)} is not a file. The table's cells "
                f"are figures copied out of a run, and a copy is held to the run it was copied "
                f"from; with no {label} there is nothing on one side of that comparison"
            )
            return lines, findings

    text = report_path.read_text(encoding="utf-8", errors="replace")
    found = classes_of(text)
    line = f"  {display(report_path)}"

    if not found:
        lines.append(line + ": no class line")
        findings.append(
            f"{display(report_path)} carries no class line this reading can place: no line of it "
            f"reads `  <precision> <shape>  possible ... fastest <entry> at <figure> "
            f"ns/argument`, which is what the probe prints for a class it measured. The table is "
            f"a transcription of those lines, so a report without them is one the table cannot be "
            f"held to at all"
        )
        return lines, findings

    lines.append(line + f": {len(found)} class(es)")

    document = doc_path.read_text(encoding="utf-8", errors="replace")
    surface = host_surface()
    rows: dict[tuple[str, str], list[dict]] = {}

    for row in ROW.finditer(document):
        named = ENTRY.findall(row.group("question"))

        if not named:
            continue

        entry = named[-1]

        if entry not in surface:
            findings.append(
                f"a row of the table in {display(doc_path)} names `{entry}`, and the tree's "
                f"host-surface table places no entry under that name. A row whose subject cannot "
                f"be placed in a class is a row this check cannot hold to any figure: it names "
                f"an entry the library does not serve, or one the surface table and the document "
                f"spell apart"
            )
            lines.append(f"  {entry}: not a host entry this tree places")
            continue

        precision, shape = surface[entry]
        scored = found.get((precision, shape))

        if scored is None:
            findings.append(
                f"a row of the table in {display(doc_path)} is about `{entry}`, which is the "
                f"{precision} {shape} class, and {display(report_path)} measures no such class. "
                f"The row's three cells are figures of a class this run did not rank, so nothing "
                f"in the report bears them out"
            )
            lines.append(f"  {entry}: no {precision} {shape} class in the report")
            continue

        # A class's rows are kept as a list: two rows for one class are two transcriptions of one
        # set of cells, and a reading that kept the last of them would compare one and pass over
        # the other.
        rows.setdefault((precision, shape), []).append({
            "entry": entry,
            "label": row.group("label").strip().strip("`"),
            "ns": row.group("ns").strip(),
            "count": row.group("count").strip(),
            "line": document.count("\n", 0, row.start()) + 1,
        })

    for (precision, shape), scored in sorted(found.items()):
        if (precision, shape) in rows or precision != LANE:
            continue

        findings.append(
            f"the table in {display(doc_path)} carries no row for the {precision} {shape} class, "
            f"which {display(report_path)} ranks: the document's own sentence is one row per "
            f"question shape for the double lane, so a class the run measured and the table does "
            f"not print is a hole in the table. A reader comparing the two reads the missing "
            f"class as one the library does not carry"
        )
        lines.append(f"  {precision} {shape}: no row in the table")

    for (precision, shape), class_rows in sorted(rows.items()):
        scored = found[(precision, shape)]

        if len(class_rows) > 1:
            findings.append(
                f"{display(doc_path)} prints {len(class_rows)} rows for the {precision} {shape} "
                f"class (lines {', '.join(str(row['line']) for row in class_rows)}), and the "
                f"document's sentence is one row per question shape. A reader has no way to tell "
                f"which of two rows of measured cells the run bore out, and the ones that agree "
                f"give a table that reads as corroborated when it is repeated"
            )

        for row in class_rows:
            lines.append(f"  {precision} {shape}: {row['label']} {row['ns']} ns, "
                         f"{row['count']} not placed")
            compare(doc_path, report_path, precision, shape, row, scored, findings)

    if not rows:
        findings.append(
            f"{display(doc_path)} carries no table row this check can place: no line of it is a "
            f"four-cell row whose first cell names a `Boys...` entry. The table this reads is "
            f"five rows of measured cells, and a reading that finds none of them is a reading "
            f"that has been looking at a document the table has left"
        )

    return lines, findings


def compare(doc_path: pathlib.Path, report_path: pathlib.Path, precision: str, shape: str,
            row: dict, scored: dict, findings: list[str]) -> None:
    """Hold one row's three quoted cells to the report's, adding a finding per cell that moved."""
    where = (f"the {precision} {shape} row of {display(doc_path)} at line {row['line']}")

    if row["label"] != scored["label"]:
        findings.append(
            f"{where} names `{row['label']}` as the class's fastest entry, and "
            f"{display(report_path)} ranks `{scored['label']}` first: the document's row "
            f"names an option this run did not place at the head of that class. A reader who "
            f"takes the document's answer for what the probe measured will buy an option the "
            f"run did not measure as fastest"
        )

    if scored["ns"] is not None and not same_number(row["ns"], scored["ns"]):
        findings.append(
            f"{where} states {row['ns']} ns per argument, and {display(report_path)} records "
            f"{scored['ns']} ns for that class's fastest entry: the figure the document "
            f"quotes is not the figure the run took. A nanosecond figure is a measurement of "
            f"one host, one build and one run, so the two cells are either one figure or the "
            f"document is stating a measurement nothing made"
        )

    if scored["count"] is None:
        findings.append(
            f"{where} states that {row['count']} option(s) could not be placed behind the "
            f"class's fastest entry, and {display(report_path)} states no such count for that "
            f"class. The count is what the document's own prose reads as the answer - a "
            f"leader alone or a tied group - so a row carrying one the report does not state "
            f"is the document answering a question the run did not"
        )
    elif not same_number(row["count"], scored["count"]):
        findings.append(
            f"{where} states that {row['count']} option(s) could not be placed behind the "
            f"class's fastest entry, and {display(report_path)} records {scored['count']}: "
            f"the tie the document reports is not the tie the run found, and the sentence "
            f"around the table reads the count as the answer to how much the leader is worth"
        )


def mutate(line: str, cell: int, value: str) -> str:
    """A table row with one cell's contents replaced, its spacing kept."""
    parts = line.split("|")

    if len(parts) != 6:
        raise ValueError(f"not a four-cell row: {line!r}")

    parts[cell] = f" {value} "
    return "|".join(parts)


def control() -> int:
    """Plant the three ways a cell of the table can disagree with the report.

    Each fixture is the committed document with one cell of one row replaced, so a fixture
    differs from the document in that cell and in nothing else. A control that passed would be a
    reading that cannot see the defect it exists for, which is how a figure of the table moved
    while every check over the document stayed green.
    """
    if not DOC.is_file() or not REPORT.is_file():
        print("control: the document or the report is not a file, so no fixture can be built")
        return 1

    document = DOC.read_text(encoding="utf-8", errors="replace")
    found = classes_of(REPORT.read_text(encoding="utf-8", errors="replace"))
    single = found.get((LANE, "single"))
    orders = found.get((LANE, "all-orders"))

    if single is None or orders is None:
        print(f"control: the report carries no {LANE} single or all-orders class, so the "
              f"fixtures this control plants cannot be built from a real row")
        return 1

    row_for: dict[str, str] = {}

    for row in ROW.finditer(document):
        named = ENTRY.findall(row.group("question"))

        if named:
            row_for.setdefault(named[-1], row.group(0))

    single_row = next((line for line in row_for.values() if single["label"] in line), "")
    orders_row = next((line for line in row_for.values() if orders["label"] in line), "")

    if not single_row or not orders_row:
        print(f"control: the document carries no row naming `{single['label']}` or "
              f"`{orders['label']}`, so the fixtures this control plants cannot be built")
        return 1

    arms = [
        ("a figure of the all-orders row moved off what the run took",
         orders_row, mutate(orders_row, NS, f"{float(orders['ns']) + 11.33:.2f}"),
         "is not the figure the run took"),
        ("the single row's entry replaced by this run's all-orders entry",
         single_row, mutate(single_row, LABEL, orders["label"]),
         "did not place at the head of that class"),
        ("a count of the single row moved off what the run found",
         single_row, mutate(single_row, COUNT, str(int(single["count"]) + 1)),
         "not the tie the run found"),
        ("the all-orders row printed twice",
         orders_row, orders_row + "\n" + orders_row,
         "rows for the fp64 all-orders class"),
    ]

    WORK.mkdir(parents=True, exist_ok=True)
    failures = 0

    for index, (what, original, planted, expected) in enumerate(arms):
        fixture = WORK / f"control-{index}.md"
        fixture.write_text(document.replace(original, planted, 1), encoding="utf-8")
        _, findings = read(fixture, REPORT)
        caught = any(expected in finding for finding in findings)

        print(f"control {index}: {what} - written to {display(fixture)}")
        print(f"  {'caught' if caught else 'NOT CAUGHT'}"
              + ("" if caught else f": no finding reads `{expected}`"))
        failures += 0 if caught else 1

    if failures:
        print(f"\ncontrol: {failures} of {len(arms)} planted defect(s) were not caught. This "
              f"check cannot see the defect it exists for, so a green run of it says nothing")
        return 1

    print(f"\ncontrol: all {len(arms)} planted defect(s) caught")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", default=str(DOC),
                        help="the document carrying the measured table")
    parser.add_argument("--report", default=str(REPORT),
                        help="the option probe's recorded report")
    parser.add_argument("--control", action="store_true",
                        help="read a fixture planted in a scratch copy instead of the document, "
                             "and exit non-zero unless every planted defect is caught")
    args = parser.parse_args()

    if args.control:
        return control()

    lines, findings = read(pathlib.Path(args.doc), pathlib.Path(args.report))

    for line in lines:
        print(line)

    if findings:
        print(f"\n{len(findings)} finding(s):\n")
        for finding in findings:
            print(f"MISMATCH: {finding}\n")
        return 1

    print(
        f"\nverdict: clean - every cell of the table agrees with the report, and every {LANE} "
        f"class the report ranks has its row"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
