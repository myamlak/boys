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

THE SAME SECTION STATES THE RUN'S FIGURES IN PROSE, and they are transcriptions of the same report:
the machine the run was taken on, the arithmetic the build it was taken from leaves, how much of
the space it served and measured, its workload, the bound the quoted entries document, and two of
the table's own readings said again in a sentence. A table held while the sentence around it is
not is half a reading: this report is going to be re-made, and a figure that moves under a
sentence moves out of agreement with nothing at all. So each of those sentences is read too,
against the report cell it is about; and the section's numerals are read as a set, so one the
report carries no cell for is reported rather than passed over - a figure no run recorded is the
defect, and reading past it is the blind spot this check exists to close.

What the check does not read, and why:
  * it does not judge the report. Whether the run that report records is this tree's own is
    `tools/check_probe_run.py`'s claim, and the two fail apart so that a stale record and a
    mismatched transcription are not reported as one thing;
  * it does not read the rest of the document. The runs the guide quotes beside its programs are
    held to those programs by `tools/check_example_transcripts.py`, and the page's other prose by
    `tools/check_doc_arithmetic.py`. Nothing here is an opinion about figures outside this
    section.

Usage:
    python3 tools/check_probe_table.py
    python3 tools/check_probe_table.py --doc build-probe-table/doc.md --report some/report.txt
    python3 tools/check_probe_table.py --control

Exit status is 0 when every cell of the table agrees with the report, every figure the section
states in prose agrees with the cell it is about, and every double-lane class the report carries
has a row, and 1 when one does not, when the table or the report cannot be read, when the document
carries no row this check can place, or when a row names an entry the tree's own host-surface
table does not place in a class. A run with `--control` exits 0 when every planted defect is
caught and 1 when one is not.
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
# entry that run ranked first, the figure that entry took and the error bound the line documents.
# The rule is the one the table was extracted by, which reproduces the committed table cell for
# cell.
CLASS_LINE = re.compile(
    r"^  (?P<precision>fp64|fp32|fp16|bf16) (?P<shape>[a-z-]+)\s+possible .*?\| "
    r"(?P<measured>\d+) measured \| fastest "
    r"(?P<label>\S+) at (?P<ns>[0-9.]+) ns/argument, documented at (?P<bound>\S+)",
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

# The section the table sits in, found by the anchor its heading carries rather than by its words:
# the words are a title someone may rewrite, and the anchor is the link other pages arrive through.
SECTION = re.compile(r"^## [^\n]*\{#what-the-double-lane-costs\}[^\n]*$")
HEADING = re.compile(r"^## ")

# The report's cells the section's prose states in sentences, each anchored on the report's own
# label. What is written here is the label and the wording around it; the figures are read out of
# the report at run time, so a re-made report moves them and this file does not.
FACT = {
    "machine": re.compile(
        r"machine:\s*(?P<cpus>\d+)\s+logical processors\s*\|\s*(?P<tier>\S+?) tier:\s*"
        r"(?P<state>\S+)"),
    "backends": re.compile(
        r"is a single rounding here:\s*(?P<rows>(?:[a-z0-9-]+\s+contracts=\S+\s*)+)"),
    "workload": re.compile(
        r"workload:\s*(?P<arguments>\d+)\s+arguments\s*\|\s*x log-uniform on\s*\[\s*(?P<lo>[^,\s]+)\s*,"
        r"\s*(?P<hi>[^\]\s]+)\s*\]\s*\|\s*each argument's own highest order drawn as the sum of two"
        r"\s+shell angular momenta over\s*(?P<order_lo>\d+)\.\.(?P<order_hi>\d+)\s*\(nmax\s*(?P<nmax>\d+)\)"),
    "totals": re.compile(
        r"the class-by-class sums:\s*(?P<possible>\d+)\s+possible\s*=\s*(?P<served>\d+)\s+served\s*"
        r"\+\s*(?P<refused>\d+)\s+refused,.*?and\s*(?P<measured>\d+)\s+cell\(s\) measured"),
}

# The sentences of that section which state those cells, each as the section writes it with the
# figure it states captured. The reading below and the control both use these, so a control that
# plants a defect plants it in the sentence the reading reads and not in a paraphrase of it.
SENTENCE = {
    "machine": re.compile(
        r"(?P<cpus>[\d,]+) logical processors with the (?P<tier>[A-Za-z0-9+]+) tier "
        r"(?P<state>[a-z]+)"),
    "backends": re.compile(
        r"(?P<count>[a-z]+) arithmetic backends\b.{0,240}?each leave a bare `a \* b \+ c` as "
        r"(?P<roundings>[a-z]+) roundings"),
    "workload": re.compile(
        r"Its workload is (?P<arguments>[\d,]+) arguments, x log-uniform over "
        r"\[(?P<lo>[^,\]]+), (?P<hi>[^\]]+)\], each argument's own highest order drawn as the sum "
        r"of two shell angular momenta over (?P<order_lo>\d+)\.\.(?P<order_hi>\d+), up to nmax "
        r"(?P<nmax>\d+)"),
    "totals": re.compile(
        r"It served every combination its classes hold, (?P<possible>[\d,]+) of them and "
        r"(?P<refused>[a-z0-9,]+) refusals, and measured (?P<measured>[\d,]+) on this host"),
    "bound": re.compile(
        r"each the fastest entry of its class and all of them documented at "
        r"(?P<bound>[0-9][0-9.eE+-]*)"),
    "separated": re.compile(
        r"(?P<ns>[\d.]+) ns per argument is the head of a class whose next (?P<count>[a-z]+) "
        r"entries the run could not separate from it, and (?P<ns2>[\d.]+) the head of one whose "
        r"next (?P<count2>[a-z]+) it could not"),
    "choice": re.compile(
        r"the head of (?P<measured>[\d,]+) measured combinations, with (?P<count>\d+) the run "
        r"could not place behind it"),
}

# A backend row of the report's block, and the words for the counts the section spells rather than
# prints. The report asks whether a bare `a * b + c` is a single rounding; the section says how
# many roundings it leaves.
BACKEND_ROW = re.compile(r"(?P<name>[a-z0-9-]+)\s+contracts=(?P<contracts>\S+)")
ROUNDINGS = {"yes": "one", "no": "two"}
COUNT_WORD = {"no": 0, "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
              "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
              "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
              "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20}

# A numeral, as the section prints one, for the sweep over what the report has no cell for.
NUMERAL = re.compile(r"\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?")

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
    """The report's classes, as (precision, shape) -> the cells the section quotes of them.

    The cells are the three a row of the table quotes - the fastest entry, its figure and the
    count behind it - and the two the section's prose states of them, which are how many
    combinations the class measured and the bound its entries are documented at. `count` is None
    where the block states no count, which is a report this reading cannot place a row against
    rather than a count of zero.
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
            "measured": header.group("measured"),
            "bound": header.group("bound"),
        }

    return found


def same_number(one: str, other: str) -> bool:
    """Whether two cells state one figure: compared as numbers where both are, as text otherwise."""
    try:
        return float(one) == float(other)
    except ValueError:
        return one.strip() == other.strip()


def squashed(text: str) -> str:
    """A text as one line, so a sentence the document wraps still reads as one sentence."""
    return re.sub(r"\s+", " ", text)


def section_of(document: str) -> tuple[str, int]:
    """The double-lane section's prose, and the line it begins at.

    The table's rows are left out of it: the rows are read cell by cell, and a row read a second
    time as prose would report one defect twice.
    """
    lines = document.splitlines()
    start = next((index for index, line in enumerate(lines) if SECTION.match(line)), None)

    if start is None:
        return "", 0

    end = next((index for index in range(start + 1, len(lines)) if HEADING.match(lines[index])),
               len(lines))

    return "\n".join(line for line in lines[start + 1:end]
                     if not line.lstrip().startswith("|")), start + 1


def value_of(numeral: str) -> float | None:
    """The value a numeral states, or None where it states none."""
    try:
        return float(numeral.replace(",", ""))
    except ValueError:
        return None


def figures_in(text: str) -> set[float]:
    """Every figure a text states, as values: two spellings of one figure are one figure."""
    return {value for value in (value_of(match.group(0)) for match in NUMERAL.finditer(text))
            if value is not None}


def compare_prose(doc_path: pathlib.Path, report_path: pathlib.Path, document: str, report: str,
                  classes: dict[tuple[str, str], dict], rows: dict[tuple[str, str], list[dict]],
                  findings: list[str], lines: list[str]) -> None:
    """Hold the figures the section states in sentences to the report's own cells.

    Each reading below is a phrase of the section's and the report's cell the phrase is about: the
    phrase is written here, the figure is read out of the report. A phrase a reading needs and the
    section does not carry is a finding as well, because a sentence and the reading of it are one
    thing: a reading whose sentence is absent is a check over nothing.

    The section's numerals are then read as a set: one the report carries no cell for is a figure
    no run recorded, which is the defect no comparison of cell against cell can find, because
    there is no second cell to compare it with.
    """
    prose, start = section_of(document)
    where = f"the double-lane section of {display(doc_path)}"

    if not prose.strip():
        findings.append(
            f"{where} is not in the document: no heading of {display(doc_path)} carries the "
            f"anchor `#what-the-double-lane-costs`. That section is what this reading is over - "
            f"the figures it states in a sentence are as much a transcription of the report as "
            f"the table's cells are - so a document that no longer carries it is one whose "
            f"figures this check has stopped reading"
        )
        return

    said, header = squashed(prose), squashed(report)
    lines.append(f"  the section's prose, from line {start}: held to the report's own header")

    def sentence(fact: str) -> tuple[re.Match | None, re.Match | None]:
        """The report's cell for a fact and the section's sentence about it, each or None."""
        recorded = FACT[fact].search(header)
        stated = SENTENCE[fact].search(said)

        if recorded is None:
            findings.append(
                f"{where} states a figure of the run, and {display(report_path)} carries no cell "
                f"for it: the report has no line this reading can take for {fact}. A report that "
                f"stopped stating it leaves the sentence about it held to nothing, which reads "
                f"exactly like a sentence that is right"
            )

        if stated is None:
            findings.append(
                f"{where} no longer states {fact}: the phrase `{SENTENCE[fact].pattern}` is not "
                f"in the section. That sentence is how this check holds the figure, and a "
                f"sentence rewritten out of the document takes its reading with it - a reading "
                f"left behind is a check over nothing"
            )

        return recorded, stated

    def moved(what: str, stated: str, recorded: str, why: str) -> None:
        """The finding for one figure the section states and the report states otherwise."""
        if not same_number(stated, recorded):
            findings.append(
                f"{where} states {stated} for {what}, and {display(report_path)} records "
                f"{recorded}. {why}"
            )

    recorded, stated = sentence("machine")

    if recorded is not None and stated is not None:
        moved("the processors the run was taken on", stated.group("cpus").replace(",", ""),
              recorded.group("cpus"),
              "A reader takes from this figure how much machine the timings came from, and it is "
              "the first cell that moves when the report is re-made on another host")
        moved("the instruction tier the run's machine carries", stated.group("tier"),
              recorded.group("tier"),
              "The tier is why the entries the table ranks are the ones they are - an entry "
              "written for another tier is not the entry this run measured - so a sentence "
              "naming a different tier quotes a different machine")
        moved("whether that tier is present on it", stated.group("state"), recorded.group("state"),
              "A tier the run reports as absent is a machine the entries were chosen around "
              "rather than written for, which is the one thing a reader of the table cannot see "
              "in the figures themselves")
        lines.append(f"  machine: {stated.group('cpus')} logical processors, "
                     f"{stated.group('tier')} tier {stated.group('state')}")

    recorded, stated = sentence("backends")

    if recorded is not None and stated is not None:
        backends = BACKEND_ROW.findall(recorded.group("rows"))
        unplaced = [name for name, contracts in backends if contracts not in ROUNDINGS]
        leaves = {ROUNDINGS[contracts] for _, contracts in backends if contracts in ROUNDINGS}
        count = COUNT_WORD.get(stated.group("count"))

        if count is None:
            findings.append(
                f"{where} counts the build's arithmetic backends as "
                f"`{stated.group('count')}`, which is not a count this reading can place. The "
                f"count is read from the backends {display(report_path)} lists, so a sentence "
                f"that counts them in other words is a sentence about a build this check cannot "
                f"hold"
            )
        elif count != len(backends):
            findings.append(
                f"{where} states that the build carries {stated.group('count')} arithmetic "
                f"backends, and {display(report_path)} lists {len(backends)} of them: "
                f"{', '.join(name for name, _ in backends)}. A backend added or dropped is a "
                f"build that measures and serves a different surface, so a count that moved is a "
                f"sentence quoting another build"
            )

        if unplaced:
            findings.append(
                f"{where} states what the build's arithmetic leaves, and "
                f"{display(report_path)} carries a contracts value this reading cannot place: "
                f"{', '.join(f'{name} {contracts}' for name, contracts in backends)}. The "
                f"report's own sentence is that the column says whether a bare `a * b + c` is a "
                f"single rounding, so a value outside yes/no is a report this reading does not "
                f"follow"
            )
        elif len(leaves) > 1:
            findings.append(
                f"{where} states one reading of what a bare multiply-add leaves, and the "
                f"backends in {display(report_path)} do not agree on one: "
                f"{', '.join(f'{name} {contracts}' for name, contracts in backends)}. One "
                f"rounding or two is a fact of the build, and a build whose backends differ is "
                f"one the sentence beside the table cannot be true of"
            )
        elif leaves and stated.group("roundings") not in leaves:
            findings.append(
                f"{where} states that each backend leaves a bare `a * b + c` as "
                f"{stated.group('roundings')} roundings, and {display(report_path)} records "
                f"{', '.join(f'{name} {contracts}' for name, contracts in backends)}, which is "
                f"{'/'.join(sorted(leaves))} rounding(s). What the arithmetic leaves is why two "
                f"entries of one class can agree on a figure and differ in accuracy, so a "
                f"sentence that moves it describes a build this run was not taken on"
            )
        lines.append(f"  arithmetic: {stated.group('count')} backend(s), each leaving a bare "
                     f"multiply-add as {stated.group('roundings')} roundings")

    recorded, stated = sentence("workload")

    if recorded is not None and stated is not None:
        for what, one, other in (
            ("the argument count of the workload it quotes",
             stated.group("arguments").replace(",", ""), recorded.group("arguments")),
            ("the low end of that workload's log-uniform range", stated.group("lo"),
             recorded.group("lo")),
            ("the high end of that range", stated.group("hi"), recorded.group("hi")),
            ("the lowest order the workload draws", stated.group("order_lo"),
             recorded.group("order_lo")),
            ("the highest order it draws", stated.group("order_hi"), recorded.group("order_hi")),
            ("the nmax it draws up to", stated.group("nmax"), recorded.group("nmax")),
        ):
            moved(what, one, other,
                  "Every figure beside it is a rate over this workload, so a reader who has the "
                  "wrong arguments, orders or range cannot tell what the figures are per unit of")

        lines.append(f"  workload: {stated.group('arguments')} arguments, x log-uniform on "
                     f"[{stated.group('lo')}, {stated.group('hi')}], orders "
                     f"{stated.group('order_lo')}..{stated.group('order_hi')}, nmax "
                     f"{stated.group('nmax')}")

    recorded, stated = sentence("totals")

    if recorded is not None and stated is not None:
        moved("the combinations the classes hold", stated.group("possible").replace(",", ""),
              recorded.group("possible"),
              "This is the document's answer to how much of the option space the run covered, "
              "and a reader takes the coverage of the table's rows from it")
        moved("what the run measured on this host", stated.group("measured").replace(",", ""),
              recorded.group("measured"),
              "The measured cell is how many combinations the figures beside it were taken "
              "over, so a figure that moves here moves the weight of the whole table")

        refused = COUNT_WORD.get(stated.group("refused"))

        if refused is None:
            findings.append(
                f"{where} states the run's refusals as `{stated.group('refused')}`, which is not "
                f"a count this reading can place, and {display(report_path)} records "
                f"{recorded.group('refused')} refused. Whether the run refused anything is part "
                f"of what its coverage claim means"
            )
        elif not same_number(str(refused), recorded.group("refused")):
            findings.append(
                f"{where} states {stated.group('refused')} refusals, and "
                f"{display(report_path)} records {recorded.group('refused')} refused. A refusal "
                f"is a combination the space holds that no entry answered, so the sentence "
                f"beside the table is where a reader learns whether any combination went "
                f"unanswered"
            )
        lines.append(f"  coverage: {stated.group('possible')} combinations the classes hold, "
                     f"{stated.group('refused')} refusals, {stated.group('measured')} measured")

    stated = SENTENCE["bound"].search(said)

    if stated is None:
        findings.append(
            f"{where} no longer states the bound its rows document: the phrase "
            f"`{SENTENCE['bound'].pattern}` is not in the section. The bound is what says how "
            f"far the entry a row quotes can be trusted, and a sentence dropped without the "
            f"reading of it leaves this check holding figures whose accuracy nothing states"
        )
    else:
        for key in sorted(rows):
            bound = classes[key]["bound"]

            if not same_number(stated.group("bound"), bound):
                findings.append(
                    f"{where} states that every row's entry is documented at "
                    f"{stated.group('bound')}, and the {key[0]} {key[1]} row of the table quotes "
                    f"an entry {display(report_path)} documents at {bound}. A bound is the "
                    f"accuracy the run proved its entry to, so a sentence stating one bound for "
                    f"rows that document another sells the rows at an accuracy the run did not "
                    f"measure"
                )

        lines.append(f"  documented bound: {stated.group('bound')} for every row quoted")

    stated = SENTENCE["separated"].search(said)

    if stated is None:
        findings.append(
            f"{where} no longer reads the table's last column back: the phrase "
            f"`{SENTENCE['separated'].pattern}` is not in the section. That sentence is the "
            f"section's claim about what the counts mean - a leader alone or a tied group - so "
            f"dropping it takes the reading of the column out of the document"
        )
    else:
        for ns, word in ((stated.group("ns"), stated.group("count")),
                         (stated.group("ns2"), stated.group("count2"))):
            behind = [key for key, scored in classes.items() if same_number(ns, scored["ns"])]
            count = COUNT_WORD.get(word)

            if count is None:
                findings.append(
                    f"{where} states that the run could not separate the next `{word}` entries "
                    f"from the head of a class, which is not a count this reading can place: the "
                    f"count is read from the report's own sentence under that class"
                )
            elif not behind:
                findings.append(
                    f"{where} states that {ns} ns per argument is the head of a class of the "
                    f"{LANE} lane, and {display(report_path)} ranks no class at that figure. The "
                    f"sentence is the document reading its own table's last column, so a figure "
                    f"no class took is a reading of a column the report does not carry"
                )
            else:
                for key in behind:
                    held = classes[key]["count"]

                    if held is None or not same_number(str(count), held):
                        findings.append(
                            f"{where} states that {ns} ns per argument heads a class whose next "
                            f"{word} entries the run could not separate from it, and "
                            f"{display(report_path)} states {held} option(s) it could not place "
                            f"behind that class's fastest entry. The count is the length of the "
                            f"tail behind the leader, which is what carries from one host to "
                            f"another, so the two are either one tail or the sentence reads the "
                            f"column as longer or shorter than the run found it"
                        )
                    else:
                        lines.append(f"  tail: {ns} ns per argument, {word} of the {key[0]} "
                                     f"{key[1]} class's entries behind it")

    stated = SENTENCE["choice"].search(said)

    if stated is None:
        findings.append(
            f"{where} no longer says what one of its rows heads: the phrase "
            f"`{SENTENCE['choice'].pattern}` is not in the section. That sentence is how the "
            f"section tells a reader what the batch row's figure is a figure of, and a reading "
            f"kept after it went is a check over nothing"
        )
    else:
        measured, count = stated.group("measured").replace(",", ""), stated.group("count")
        held = [key for key, scored in classes.items()
                if same_number(measured, scored["measured"])
                and scored["count"] is not None and same_number(count, scored["count"])]

        if not held:
            findings.append(
                f"{where} states that one of its rows heads {measured} measured combinations, "
                f"with {count} the run could not place behind it, and no class of "
                f"{display(report_path)} carries that pair. The {LANE} classes this report ranks "
                f"hold "
                + ", ".join(f"{key[1]} {scored['measured']} measured and {scored['count']} "
                            f"unplaced" for key, scored in sorted(classes.items())
                            if key[0] == LANE)
                + ". A pair no class carries is a figure of a run that was never made"
            )
        else:
            lines.append(f"  the batch row: the head of {measured} measured combinations, "
                         f"{count} unplaced - the {', '.join(key[1] for key in held)} class(es)")

    recorded_figures = figures_in(report)
    stated_figures: set[float] = set()
    loose: list[re.Match] = []

    for numeral in NUMERAL.finditer(said):
        value = value_of(numeral.group(0))

        if value is None:
            continue

        stated_figures.add(value)

        if value not in recorded_figures:
            loose.append(numeral)

    lines.append(f"  the section's figures: {len(stated_figures)} distinct numeral(s) stated, "
                 f"{len(loose)} of them in no cell of the report")

    for numeral in loose:
        around = said[max(0, numeral.start() - 60):numeral.end() + 40]

        findings.append(
            f"{where} states the figure {numeral.group(0)}, and {display(report_path)} states no "
            f"such figure anywhere in it: `...{around}...`. A figure no run recorded is the one "
            f"wrong number no comparison of a quoted cell against its own cell can find, because "
            f"there is no cell to compare it with - so it is reported here rather than passed "
            f"over"
        )


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

    compare_prose(doc_path, report_path, document, text, found, rows, findings, lines)

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
    """Plant the ways a figure of the section can disagree with the report.

    A fixture is the committed document with one cell of one row, or one figure of one sentence,
    replaced, so it differs from the document in that place and in nothing else. Every planted
    figure is built from the report's own cells, which is the reading the check makes. A control
    that passed would be a reading that cannot see the defect it exists for, which is how a figure
    of the table moved while every check over the document stayed green.
    """
    if not DOC.is_file() or not REPORT.is_file():
        print("control: the document or the report is not a file, so no fixture can be built")
        return 1

    document = DOC.read_text(encoding="utf-8", errors="replace")
    report = REPORT.read_text(encoding="utf-8", errors="replace")
    found = classes_of(report)
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
         document.replace(orders_row, mutate(orders_row, NS, f"{float(orders['ns']) + 11.33:.2f}"),
                         1),
         "is not the figure the run took"),
        ("the single row's entry replaced by this run's all-orders entry",
         document.replace(single_row, mutate(single_row, LABEL, orders["label"]), 1),
         "did not place at the head of that class"),
        ("a count of the single row moved off what the run found",
         document.replace(single_row, mutate(single_row, COUNT, str(int(single["count"]) + 1)),
                          1),
         "not the tie the run found"),
        ("the all-orders row printed twice",
         document.replace(orders_row, orders_row + "\n" + orders_row, 1),
         "rows for the fp64 all-orders class"),
    ]

    header = squashed(report)
    machine, workload = FACT["machine"].search(header), FACT["workload"].search(header)
    totals = FACT["totals"].search(header)
    bound = found.get((LANE, "single"), {}).get("bound")

    # The sentences are located in the document as it wraps them there, so a fixture is the
    # document with one figure of one sentence moved and not a document with a sentence rewritten.
    opening = {fact: re.compile(SENTENCE[fact].pattern.replace(" ", r"\s+")).search(document)
               for fact in SENTENCE}

    if machine is None or workload is None or totals is None or bound is None \
            or None in opening.values():
        print("control: the report's header or the section's sentences no longer carry the cells "
              "the fixtures are planted from, so the figures moved in them cannot be built out of "
              "a real run")
        return 1

    unplantable: list[str] = []

    def arm(what: str, fact: str, expected: str, cell: str, value: str) -> tuple | None:
        """One fixture arm: a sentence of the section with one figure of it moved.

        What the figure is moved to is built from the report's own cells, so a fixture is a
        disagreement between two figures this repository holds rather than one written here.
        """
        match = opening[fact]
        stated = match.group(cell)

        if stated is None or match.start(cell) < 0:
            unplantable.append(f"{what}: the sentence states no {cell}")
            return None

        planted = document[:match.start(cell)] + value + document[match.end(cell):]

        if planted == document:
            unplantable.append(f"{what}: moving the figure left the document as it was")
            return None

        return what, planted, expected

    prose = [candidate for candidate in (
        arm("the processors the section states moved off the machine the run was taken on",
            "machine", f"records {machine.group('cpus')}", "cpus",
            str(int(machine.group("cpus")) + 1)),
        arm("the tier the section names moved off the one the run's machine carries",
            "machine", "the instruction tier the run's machine carries", "tier",
            "SSE2" if machine.group("tier") != "SSE2" else "AVX512"),
        arm("the combinations the section says its classes hold moved up",
            "totals", "the combinations the classes hold", "possible",
            str(int(totals.group("possible")) + 1)),
        arm("what the section says the run measured moved up",
            "totals", "what the run measured on this host", "measured",
            str(int(totals.group("measured")) + 11)),
        arm("a refusal the section states where the run refused nothing",
            "totals", f"records {totals.group('refused')} refused", "refused", "two"),
        arm("the argument count of the workload halved",
            "workload", "the argument count of the workload it quotes", "arguments",
            str(int(workload.group("arguments")) // 2)),
        arm("the bound the section documents its rows at moved off the run's",
            "bound", f"documents at {bound}", "bound",
            next((scored["bound"] for _, scored in sorted(found.items())
                  if scored["bound"] != bound), "0")),
        arm("a tail length the section reads back moved",
            "separated", "the length of the tail behind the leader", "count",
            next(word for word in ("four", "five", "six", "seven")
                 if COUNT_WORD[word] != COUNT_WORD[opening["separated"].group("count")])),
        arm("the measured combinations the batch row is said to head moved up",
            "choice", "A pair no class carries", "measured",
            str(int(opening["choice"].group("measured").replace(",", "")) + 1)),
    ) if candidate is not None]

    absent = 1234567

    while absent in figures_in(report):
        absent += 1

    end = opening["machine"].end()
    prose.append(("a figure no run recorded stated beside the table",
                  f"{document[:end]} The write-down itself cost {absent} ns.{document[end:]}",
                  "states no such figure anywhere in it"))

    arms += prose

    WORK.mkdir(parents=True, exist_ok=True)
    failures = 0

    for unwritten in unplantable:
        print(f"control: could not plant {unwritten}")

    for index, (what, planted, expected) in enumerate(arms):
        fixture = WORK / f"control-{index}.md"
        fixture.write_text(planted, encoding="utf-8")
        _, findings = read(fixture, REPORT)
        caught = any(expected in finding for finding in findings)

        print(f"control {index}: {what} - written to {display(fixture)}")
        print(f"  {'caught' if caught else 'NOT CAUGHT'}"
              + ("" if caught else f": no finding reads `{expected}`"))
        failures += 0 if caught else 1

    if failures or unplantable:
        print(f"\ncontrol: {failures} of the {len(arms)} planted defect(s) were not caught, and "
              f"{len(unplantable)} could not be planted at all. This check cannot see a defect it "
              f"exists for, so a green run of it says nothing")
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
        f"\nverdict: clean - every cell of the table agrees with the report, every figure the "
        f"section states in prose agrees with the report's cell for it, and every {LANE} class "
        f"the report ranks has its row"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
