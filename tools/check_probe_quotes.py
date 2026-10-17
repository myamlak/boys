#!/usr/bin/env python3
"""Holds every figure the documents quote out of the option probe's recorded report.

`tests/data/boys_option_probe_report.txt` is a run of `boys-option-probe` kept as a file, and the
documents quote cells out of it: class leaders, their nanoseconds per argument, the number of
options a class could not place behind its leader, the two counts the closure states, and the
machine and workload the run was taken on. A quoted figure is a transcription rather than a
measurement: it is right when the copy is right and wrong when the copy is wrong, and only a
reading of both the document and the report can tell which one it is.

The report is going to be re-made on the final tree. When it is, every figure moves, and nothing
currently compares the documents' copies with the report's own rows: `tools/check_probe_table.py`
holds the table in `docs/getting-started.md`'s "the double lane costs" section, but the same
figures are stated again in that page's axis table, in `docs/specification.md`, and in
`docs/lane-contract.md`, and no check reads those. This one does, across every declared document.

WHAT IT READS. The report's first line (version, inputs and verdict), the machine, backend and
workload lines, every accuracy class's line and rows, and the two closure totals. A structural
piece it cannot read is a refusal, not a pass: a checker that cannot find what it holds must not
report a clean set.

THE FORMS IT ACCEPTS, AND WHY EACH IS A QUOTE OF THE REPORT:

  1. `A against B ns per argument for the fastest narrow row` - the fp64-single class statement.
     `A` is held to the class line's leader figure. `B` is held to a row of the same class whose
     name carries `-narrow-`; the sentence contrasts the uniform grid with the narrow partition,
     and the report's rational-minimax rows are the alternative route, so `B` is held to the
     fastest narrow row *on the shipped route* (the narrow rows whose names do not say
     `rational-minimax`). The sentence's `fp64 single class` must appear in the same paragraph,
     or the figures are reported rather than silently tied to it.
  2. A markdown row `| ... `BoysX` | `option-name` | ns | count |` - the getting-started table.
     The `Boys` name in the first cell selects the fp64 class; the option name is held to that
     class's leader, the figure to the leader's row, and the count to the class's "N could not be
     placed behind it" line. On the page the inventory says owns this table, every fp64 class the
     report carries must have exactly one row; a page that merely prints a row is held to that row.
  3. `all of them documented at B` - held to every fp64 class line's `documented at` cell.
  4. `A ns per argument is the head of a class whose next N entries ... and B the head of one
     whose next M ...` - the two readings repeated in prose. Each figure is tied to the single
     fp64 class whose leader figure and unplaced count both match; an ambiguous or empty tie is
     reported, never guessed.
  5. `N of them and no refusals, and measured M on this host` - `N` is held to the report's
     closure `possible`/`served` and `M` to its `measured` cell.
  6. `the head of N measured combinations, with M the run could not place behind it` - held to
     the fp64 all-n class (the batch shape the sentence names).
  7. The machine, backend and workload sentences - the logical-processor count, the tier, the
     number of backends and their rounding statement, and the workload's argument count, range,
     order range and nmax, each held to the report's own line.
  8. A transcript block of `RunOptionProbe`'s text. This is the one form whose figures the report
     does not carry: the page says the transcript is a different, older run. It is listed as a
     declared non-subject, provided the page still says so; a transcript with that sentence gone
     is a finding, because a reader would take it for the report.

WHAT IT DOES NOT CHECK, AND WHY:

  * the example transcript's figures (2018 options, 72 ranked, 44.54 ns/argument, 2 unplaced).
    They are a short protocol run on the build the page was written against, the page says its
    counts differ from the current build's, and the report carries 2,448 measured options and
    a 144-option fp64 all-orders class. It is listed as not-checked with that reason.
  * a figure computed by combining two report rows. No declared site quotes one: the pair
    statements print two rows side by side and not a ratio of them. The report's own closure sum
    (2,448 measured + 144 device cells = 2,592) is the report's arithmetic; the documents' 2,592
    and 2,448 are held to the closure's stated cells, not recomputed here.
  * the ranking behind `fastest narrow row` across the rational route as well. The check holds
    the figure to the shipped-route narrow leader and prints the report's overall narrow minimum
    as a note when the two differ; judging whether the prose should have said "shipped route" is
    a reading of the report's prose, not of its cells.
  * whether the report was taken from this tree. That is `tools/check_probe_run.py`'s claim, and
    the two fail apart so a stale record and a mismatched transcription are not one report.

IT REFUSES rather than passing when the report lacks a stamp, a class table or a closure cell;
when a declared document is absent; when a site the inventory says must exist is not found; or
when the run finds no site at all. Prose that quotes nothing and prose whose quotes are all
correct produce the same empty report, and only the second is a clean set.

Usage:
    python3 tools/check_probe_quotes.py --check
    python3 tools/check_probe_quotes.py --check --root <tree> --report <report.txt>
    python3 tools/check_probe_quotes.py --check --root <tree> --report <report> --show-agree

Exit status: 0 when every quoted figure and name agrees with the report, the declared non-subjects
are present and still declared, and every inventory site is found; 1 otherwise.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from collections import Counter
from dataclasses import dataclass, field

REPO = pathlib.Path(__file__).resolve().parent.parent
REPORT = pathlib.Path("tests/data/boys_option_probe_report.txt")

# The declared set: the documents a reader outside this repository can open, and the examples'
# comments. Every one of these is scanned; the inventory below says which of them must carry a
# site of which kind.
TOP_FILES = ("README.md", "CHANGELOG.md")
SCAN_DIRS = (("docs", "*.md"), ("examples", "*.cpp"))

# What the known sites are, per document and per kind, as a floor rather than a cap: a new quote
# found by the same patterns is held to the report too, but a known site that disappears is a
# finding, because a checker that can pass over a removed site is not holding anything.
EXPECTED = {
    "docs/getting-started.md": {
        "prose-pair": 1,        # the fp64-single 56.09 / 246.21 statement
        "machine": 1,           # logical processors and tier
        "backends": 1,          # four arithmetic backends, two roundings
        "workload": 1,          # arguments, range, orders, nmax
        "run-counts": 1,        # 2,592 served and no refusals, 2,448 measured
        "documented-at": 1,     # the five fp64 classes' bound
        "class-row": 5,         # one row per fp64 question shape
        "repeat-reading": 1,    # 29.48 / seven and 56.09 / zero
        "batch-count": 1,       # 288 combinations, 23 unplaced
        "example-transcript": 1,  # the declared different run, not checked
    },
    "docs/specification.md": {"prose-pair": 1},
    "docs/lane-contract.md": {"prose-pair": 1},
}

# The five fp64 question shapes, by the entry the document's first cell names.
ENTRY_SHAPE = {
    "BoysSingle": "single",
    "BoysAllOrders": "all-orders",
    "BoysFixedN": "fixed-n",
    "BoysAllN": "all-n",
    "BoysAllNAtOrders": "all-n-at-orders",
}

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
}


class Refusal(Exception):
    """The checker cannot find what it holds, so it refuses rather than reports a clean set."""


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------

STAMP = re.compile(
    r"^boys option probe \|.*?\| version (?P<version>\S+) \| inputs (?P<inputs>[^|]+?)"
    r" \| verdict (?P<verdict>\S+)\s*$", re.M)

MACHINE = re.compile(
    r"machine:\s*(?P<cpus>\d+)\s+logical processors\s*\|\s*(?P<tier>\S+?)\s+tier:\s*"
    r"(?P<state>\S+)")

BACKEND = re.compile(r"^\s+(?P<name>[a-z0-9-]+)\s+contracts=(?P<contracts>\S+)\s*$", re.M)

WORKLOAD = re.compile(
    r"workload:\s*(?P<arguments>\d+)\s+arguments\s*\|\s*x log-uniform on\s*\[\s*"
    r"(?P<lo>[^,\s]+)\s*,\s*(?P<hi>[^\]\s]+)\s*\]\s*\|\s*each argument's own highest\s+"
    r"order drawn as the sum of two shell angular momenta over\s*(?P<order_lo>\d+)\.\."
    r"(?P<order_hi>\d+)\s*\(nmax\s*(?P<nmax>\d+)\)", re.S)

CLASS_LINE = re.compile(
    r"^  (?P<precision>fp64|fp32|fp16|bf16) (?P<shape>[a-z-]+)\s+possible (?P<possible>\d+) = "
    r"(?P<served>\d+) served \+ (?P<refused>\d+) refused \| (?P<measured>\d+) measured \| fastest "
    r"(?P<leader>\S+) at (?P<ns>\d+\.\d+) ns/argument, documented at (?P<bound>\S+) \| "
    r"(?P<ordered>\S+)", re.M)

UNPLACED = re.compile(r"^\s+(?P<count>\d+) could not be placed behind it\b", re.M)
PARTITION = re.compile(r"partition\s+FitGranularity::(?P<partition>\w+)")
CLASS_ROW = re.compile(
    r"(?P<name>[a-z][a-z0-9-]+)[ \t]+(?P<ns>\d+\.\d+)[ \t]+ns[ \t]+\((?P<bound>[^)]+)\)")

CLOSURE = re.compile(
    r"the class-by-class sums:\s*(?P<possible>\d+)\s+possible\s*=\s*(?P<served>\d+)\s+served\s*"
    r"\+\s*(?P<refused>\d+)\s+refused,.*?and\s+(?P<measured>\d+)\s+cell\(s\) measured", re.S)

SPACE = re.compile(
    r"what this build does with each:\s*(?P<served>\d+)\s+served\s*\+\s*(?P<refused>\d+)\s+"
    r"refused\s*=\s*(?P<cells>\d+)\s+cells over\s+(?P<classes>\d+)\s+class\(es\)", re.S)


@dataclass
class ClassBlock:
    """One accuracy class, as the report's class line and block state it."""

    precision: str
    shape: str
    possible: int
    served: int
    refused: int
    measured: int
    leader: str
    leader_ns: str
    bound: str
    ordered: str
    unplaced: int = -1
    partition: str = ""
    rows: dict[str, str] = field(default_factory=dict)   # name -> ns (text)

    @property
    def key(self) -> tuple[str, str]:
        return (self.precision, self.shape)


@dataclass
class Report:
    version: str
    inputs: str
    verdict: str
    cpus: int
    tier: str
    tier_state: str
    backends: dict[str, bool]                            # name -> contracts
    workload: dict[str, str]
    classes: dict[tuple[str, str], ClassBlock]
    closure: dict[str, int]
    space: dict[str, int]

    def rows(self) -> set[str]:
        return {name for block in self.classes.values() for name in block.rows}


def load_report(path: pathlib.Path) -> Report:
    """The report's stamp, machine lines, class tables and closure, or a refusal."""
    if not path.is_file():
        raise Refusal(f"no option probe report at {path}")

    text = path.read_text(encoding="utf-8", errors="replace")

    stamp = STAMP.search(text)
    if stamp is None:
        raise Refusal(
            f"{path} carries no `boys option probe | ... | version ... | inputs ... | verdict ...` "
            f"line, so the quoted figures cannot be held to a report identity")
    machine = MACHINE.search(text)
    if machine is None:
        raise Refusal(f"{path} carries no `machine: N logical processors | TIER tier: STATE` line")
    workload = WORKLOAD.search(text)
    if workload is None:
        raise Refusal(f"{path} carries no readable `workload:` line")

    backend_start = text.find("arithmetic backends this build carries")
    workload_start = text.find("workload:", backend_start if backend_start >= 0 else 0)
    if backend_start < 0 or workload_start < 0:
        raise Refusal(f"{path} carries no `arithmetic backends ...` block beside its workload line")
    backends = {m.group("name"): m.group("contracts") == "yes"
                for m in BACKEND.finditer(text[backend_start:workload_start])}
    if not backends:
        raise Refusal(f"{path} names no arithmetic backend, so the documents' rounding statement "
                      f"has nothing to be held to")

    matches = list(CLASS_LINE.finditer(text))
    if not matches:
        raise Refusal(f"{path} carries no accuracy class line (`  fp64 single  possible ... | "
                      f"fastest ... at ... ns/argument, documented at ... | ...`)")
    section_end = text.find("\nthe option space", matches[-1].end())
    if section_end < 0:
        section_end = len(text)

    classes: dict[tuple[str, str], ClassBlock] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else section_end
        block = text[match.end():end]
        rows: dict[str, str] = {}
        for row in CLASS_ROW.finditer(block):
            rows[row.group("name")] = row.group("ns")
        if len(rows) != int(match.group("measured")):
            raise Refusal(
                f"{path}: the {match.group('precision')} {match.group('shape')} class states "
                f"{match.group('measured')} measured option(s) and {len(rows)} row(s) parse; a "
                f"class table this check cannot read whole is refused")
        if rows.get(match.group("leader")) != match.group("ns"):
            raise Refusal(
                f"{path}: the {match.group('precision')} {match.group('shape')} class names "
                f"`{match.group('leader')}` at {match.group('ns')} ns/argument and its own rows do "
                f"not carry that pair")
        unplaced = UNPLACED.search(block)
        partition = PARTITION.search(block)
        parsed = ClassBlock(
            precision=match.group("precision"), shape=match.group("shape"),
            possible=int(match.group("possible")), served=int(match.group("served")),
            refused=int(match.group("refused")), measured=int(match.group("measured")),
            leader=match.group("leader"), leader_ns=match.group("ns"),
            bound=match.group("bound"), ordered=match.group("ordered"),
            unplaced=int(unplaced.group("count")) if unplaced else -1,
            partition=partition.group("partition") if partition else "",
            rows=rows)
        if parsed.key in classes:
            raise Refusal(f"{path}: class {parsed.key} appears twice")
        if parsed.unplaced < 0:
            raise Refusal(f"{path}: the {parsed.key} class states no `N could not be placed "
                          f"behind it` line")
        classes[parsed.key] = parsed

    if len(classes) != 20:
        raise Refusal(f"{path}: {len(classes)} class(es) parse, and this probe's space is stated "
                      f"over 20; a partial class section is refused")
    if len([key for key in classes if key[0] == "fp64"]) != 5:
        raise Refusal(f"{path}: {len([key for key in classes if key[0] == 'fp64'])} fp64 class(es) "
                      f"parse, and the documents quote five question shapes")

    closure = CLOSURE.search(text)
    if closure is None:
        raise Refusal(f"{path} carries no `the class-by-class sums: N possible = N served + N "
                      f"refused, ... and M cell(s) measured` line")
    space = SPACE.search(text)
    if space is None:
        raise Refusal(f"{path} carries no `N served + N refused = N cells over N class(es)` line")

    return Report(
        version=stamp.group("version"), inputs=stamp.group("inputs").strip(),
        verdict=stamp.group("verdict"), cpus=int(machine.group("cpus")),
        tier=machine.group("tier"), tier_state=machine.group("state"),
        backends=backends,
        workload={k: workload.group(k) for k in
                  ("arguments", "lo", "hi", "order_lo", "order_hi", "nmax")},
        classes=classes,
        closure={k: int(closure.group(k)) for k in ("possible", "served", "refused", "measured")},
        space={k: int(space.group(k)) for k in ("served", "refused", "cells", "classes")},
    )


# --------------------------------------------------------------------------
# The documents' forms
# --------------------------------------------------------------------------

PAIR = re.compile(
    r"(?P<leader>\d+\.\d+)\s+against\s+(?P<narrow>\d+\.\d+)\s+ns per argument for the fastest\s+"
    r"narrow row")

DOCUMENTED = re.compile(r"all of them documented at (?P<bound>[0-9][0-9.eE+-]*)")

REPEAT = re.compile(
    r"(?P<first>\d+\.\d+)\s+ns per\s+argument is the head of a class whose next\s+"
    r"(?P<first_count>[a-z]+)\s+entries the run could not separate from it,\s+and\s+"
    r"(?P<second>\d+\.\d+)\s+the head of one whose next\s+(?P<second_count>[a-z]+)\s+"
    r"it could not")

RUN_COUNTS = re.compile(
    r"It served every combination its classes hold,\s*(?P<possible>[\d,]+)\s+of them and no "
    r"refusals, and measured\s+(?P<measured>[\d,]+)\s+on this host")

BATCH_COUNT = re.compile(
    r"the head of\s+(?P<possible>[\d,]+)\s+measured combinations,\s+with\s+(?P<count>\d+)\s+"
    r"the run could not place behind it")

MACHINE_SENTENCE = re.compile(
    r"(?P<cpus>[\d,]+)\s+logical processors with the\s+(?P<tier>[A-Za-z0-9+]+)\s+tier\s+"
    r"(?P<state>[a-z]+)")

BACKENDS_SENTENCE = re.compile(
    r"(?P<count>[A-Za-z]+)\s+arithmetic backends\b.{0,240}?each leave a bare\s+"
    r"`a \* b \+ c`\s+as\s+(?P<roundings>[A-Za-z]+)\s+roundings", re.S)

WORKLOAD_SENTENCE = re.compile(
    r"Its workload is\s+(?P<arguments>[\d,]+)\s+arguments, x log-uniform over\s+"
    r"\[(?P<lo>[^,\]]+),\s*(?P<hi>[^\]]+)\], each argument's own highest order drawn as the sum\s+"
    r"of two shell angular momenta over\s+(?P<order_lo>\d+)\.\.(?P<order_hi>\d+), up to nmax\s+"
    r"(?P<nmax>\d+)", re.S)

TRANSCRIPT = re.compile(
    r"measured\s+(?P<measured>\d+)\s+options on\s+(?P<cpus>\d+)\s+logical processors\s*\n\s*\n"
    r"\s*fp64 all-orders - (?P<ranked>\d+) options ranked\s*\n\s*fastest: (?P<name>\S+) at\s+"
    r"(?P<ns>\d+\.\d+) ns/argument\s*\n\s*the run could not separate (?P<unplaced>\d+)", re.M)

TABLE_ROW = re.compile(
    r"^\|(?P<question>[^\n|]*)\|(?P<name>[^\n|]*)\|(?P<ns>[^\n|]*)\|(?P<count>[^\n|]*)\|"
    r"[ \t]*$", re.M)

ENTRY = re.compile(r"`(?P<entry>Boys[A-Za-z0-9_]*)`")
BACKTICKED = re.compile(r"`(?P<name>[a-z][a-z0-9-]+)`")
OPTION_NAME = re.compile(r"\b[a-z][a-z0-9-]*-(?:fp64|fp32|fp16|bf16)\b")


@dataclass
class Site:
    path: str
    line: int
    kind: str
    quote: str
    verdict: str            # "agree" | "finding" | "not-checked"
    detail: str = ""


def line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def flatten(text: str) -> tuple[str, list[int]]:
    """The text with every whitespace run collapsed to one space, and where each character was.

    The documents wrap their prose at a column, so a sentence's words are separated by `\\n` on
    paper and by ` ` on the page. A pattern that spells the sentence out has to read what the
    reader sees, and the map lets a match's offset be turned back into the document's own line.
    """
    flat: list[str] = []
    index_map: list[int] = []
    offset = 0
    length = len(text)
    while offset < length:
        if text[offset].isspace():
            end = offset
            while end < length and text[end].isspace():
                end += 1
            flat.append(" ")
            index_map.append(offset)
            offset = end
        else:
            flat.append(text[offset])
            index_map.append(offset)
            offset += 1
    return "".join(flat), index_map


def number(text: str) -> int | None:
    return NUMBER_WORDS.get(text.lower())


def plain_int(text: str) -> int:
    return int(text.replace(",", ""))


def same_number(left: str, right: str) -> bool:
    try:
        return float(left) == float(right)
    except ValueError:
        return left == right


# --------------------------------------------------------------------------
# Holding each form to the report
# --------------------------------------------------------------------------

def check_pair(path: str, flat: str, index_map: list[int], original: str,
               report: Report) -> list[Site]:
    sites = []
    cls = report.classes[("fp64", "single")]
    for match in PAIR.finditer(flat):
        line = line_of(original, index_map[match.start()])
        quote = f"{match.group('leader')} / {match.group('narrow')} ns per argument"
        prefix = flat[max(0, match.start() - 400):match.start()]
        if "fp64 single class" not in prefix:
            sites.append(Site(path, line, "prose-pair", quote, "finding",
                              "the figures are not stated beside `the fp64 single class`, so this "
                              "check cannot tie them to that class; the report's fp64 single "
                              f"leader is `{cls.leader}` at {cls.leader_ns}"))
            continue

        problems = []
        if match.group("leader") != cls.leader_ns:
            problems.append(
                f"the leader figure is quoted {match.group('leader')} and the report's fp64 "
                f"single leader `{cls.leader}` took {cls.leader_ns} ns/argument")
        if cls.partition and cls.partition != "kUniform":
            problems.append(
                f"the sentence says the uniform grid leads the class, and the report's leader "
                f"combination names FitGranularity::{cls.partition}")

        narrow = [(name, ns) for name, ns in cls.rows.items() if "-narrow-" in name]
        shipped = [(name, ns) for name, ns in narrow if "rational-minimax" not in name]
        fastest = min(shipped, key=lambda pair: float(pair[1])) if shipped else None
        overall = min(narrow, key=lambda pair: float(pair[1])) if narrow else None
        carrying = [pair for pair in narrow if pair[1] == match.group("narrow")]

        if not narrow:
            problems.append("the report's fp64 single class carries no row whose name says narrow")
        elif not carrying:
            problems.append(
                f"no narrow row of the fp64 single class carries {match.group('narrow')} "
                f"ns/argument"
                + (f"; the fastest shipped-route narrow row is `{fastest[0]}` at {fastest[1]}"
                   if fastest else ""))
        elif len(carrying) > 1:
            problems.append(
                f"{match.group('narrow')} ns/argument is carried by {len(carrying)} narrow rows "
                f"({', '.join(name for name, _ns in carrying)}), so the figure cannot be tied to "
                f"one row")
        elif fastest is not None and carrying[0] != fastest:
            problems.append(
                f"the quoted narrow row `{carrying[0][0]}` took {carrying[0][1]} ns/argument, and "
                f"the report's fastest shipped-route narrow row is `{fastest[0]}` at {fastest[1]}")

        if problems:
            sites.append(Site(path, line, "prose-pair", quote, "finding", "; ".join(problems)))
        else:
            detail = (f"leader `{cls.leader}` at {cls.leader_ns} ns/argument "
                      f"(FitGranularity::{cls.partition or '?'}); fastest shipped-route narrow row "
                      f"`{fastest[0]}` at {fastest[1]} ns/argument")
            if overall is not None and overall[1] != match.group("narrow"):
                detail += (f". note: the report's fastest narrow row across both routes is "
                           f"`{overall[0]}` at {overall[1]} ns/argument")
            sites.append(Site(path, line, "prose-pair", quote, "agree", detail))
    return sites


def check_class_rows(path: str, text: str, report: Report,
                     require_table: bool) -> list[Site]:
    """The getting-started table's rows, and the page's one-row-per-class rule.

    Only the page the inventory says carries the table is held to the completeness rule; a page
    that merely prints an option name in a table is held to that name alone.
    """
    sites = []
    fp64 = {key[1]: block for key, block in report.classes.items() if key[0] == "fp64"}
    seen: dict[str, int] = Counter()

    for match in TABLE_ROW.finditer(text):
        question, name_cell = match.group("question"), match.group("name")
        entry = ENTRY.search(question)
        name = BACKTICKED.fullmatch(name_cell.strip())
        ns_cell = match.group("ns").strip()
        count_cell = match.group("count").strip()
        if entry is None or name is None or not re.fullmatch(r"\d+\.\d+", ns_cell) \
                or not re.fullmatch(r"\d+", count_cell):
            continue
        shape = ENTRY_SHAPE.get(entry.group("entry"))
        if shape is None:
            continue
        line = line_of(text, match.start())
        quote = f"{name.group('name')} / {ns_cell} / {count_cell}"
        block = fp64.get(shape)
        seen[shape] += 1
        if block is None:
            sites.append(Site(path, line, "class-row", quote, "finding",
                              f"the report carries no fp64 {shape} class"))
            continue

        problems = []
        if name.group("name") != block.leader:
            problems.append(
                f"the row names `{name.group('name')}` and the report's fp64 {shape} leader is "
                f"`{block.leader}`")
        if name.group("name") not in block.rows:
            problems.append(f"`{name.group('name')}` is no row of the report's fp64 {shape} class")
        elif block.rows[name.group("name")] != ns_cell:
            problems.append(
                f"`{name.group('name')}` took {block.rows[name.group('name')]} ns/argument in the "
                f"report and {ns_cell} here")
        if count_cell != str(block.unplaced):
            problems.append(
                f"the run could not place {block.unplaced} option(s) behind the leader in the "
                f"report and the row says {count_cell}")
        if problems:
            sites.append(Site(path, line, "class-row", quote, "finding", "; ".join(problems)))
        else:
            sites.append(Site(path, line, "class-row", quote, "agree",
                              f"fp64 {shape} leader `{block.leader}` at {block.leader_ns}, "
                              f"{block.unplaced} unplaced"))

    if not require_table:
        return sites

    for shape, block in sorted(fp64.items()):
        if seen[shape] == 0:
            sites.append(Site(path, 0, "class-row", f"fp64 {shape}", "finding",
                              f"the report carries the fp64 {shape} class and no row of the "
                              f"document's table names it; a class with no row is a hole in the "
                              f"table, not a class left out on purpose"))
        elif seen[shape] > 1:
            sites.append(Site(path, 0, "class-row", f"fp64 {shape}", "finding",
                              f"{seen[shape]} rows name the fp64 {shape} class and the table is "
                              f"one row per question shape"))
    return sites


def check_documented(path: str, flat: str, index_map: list[int], original: str,
                     report: Report) -> list[Site]:
    sites = []
    fp64 = {key: block for key, block in report.classes.items() if key[0] == "fp64"}
    for match in DOCUMENTED.finditer(flat):
        line = line_of(original, index_map[match.start()])
        quote = match.group("bound")
        differing = [block for block in fp64.values() if block.bound != quote]
        if differing:
            detail = "; ".join(
                f"fp64 {block.shape} documents {block.bound}" for block in sorted(
                    differing, key=lambda b: b.shape))
            sites.append(Site(path, line, "documented-at", quote, "finding",
                              f"all five fp64 classes must document {quote}; {detail}"))
        else:
            sites.append(Site(path, line, "documented-at", quote, "agree",
                              "all five fp64 classes document this bound"))
    return sites


def check_repeat(path: str, flat: str, index_map: list[int], original: str,
                 report: Report) -> list[Site]:
    sites = []
    fp64 = {key: block for key, block in report.classes.items() if key[0] == "fp64"}
    for match in REPEAT.finditer(flat):
        line = line_of(original, index_map[match.start()])
        quote = (f"{match.group('first')} / {match.group('first_count')}, "
                 f"{match.group('second')} / {match.group('second_count')}")
        problems = []
        for figure, count_word in ((match.group("first"), match.group("first_count")),
                                   (match.group("second"), match.group("second_count"))):
            wanted = number(count_word)
            if wanted is None:
                problems.append(f"`{count_word}` is not a count this check reads")
                continue
            candidates = [block for block in fp64.values()
                          if block.leader_ns == figure and block.unplaced == wanted]
            if len(candidates) != 1:
                names = ", ".join(f"fp64 {block.shape}" for block in candidates) or "none"
                problems.append(
                    f"{figure} ns/argument with {count_word} unplaced ties to {names} in the "
                    f"report; the figure cannot be tied to a single class")
        if problems:
            sites.append(Site(path, line, "repeat-reading", quote, "finding", "; ".join(problems)))
        else:
            first = [b for b in fp64.values()
                     if b.leader_ns == match.group("first")
                     and b.unplaced == number(match.group("first_count"))][0]
            second = [b for b in fp64.values()
                      if b.leader_ns == match.group("second")
                      and b.unplaced == number(match.group("second_count"))][0]
            sites.append(Site(path, line, "repeat-reading", quote, "agree",
                              f"fp64 {first.shape} leader at {first.leader_ns} with "
                              f"{first.unplaced} unplaced; fp64 {second.shape} leader at "
                              f"{second.leader_ns} with {second.unplaced} unplaced"))
    return sites


def check_run_counts(path: str, flat: str, index_map: list[int], original: str,
                     report: Report) -> list[Site]:
    sites = []
    for match in RUN_COUNTS.finditer(flat):
        line = line_of(original, index_map[match.start()])
        possible = plain_int(match.group("possible"))
        measured = plain_int(match.group("measured"))
        quote = f"{possible} served / {measured} measured"
        problems = []
        if possible != report.closure["possible"] or possible != report.closure["served"]:
            problems.append(
                f"the report's closure states {report.closure['possible']} possible = "
                f"{report.closure['served']} served")
        if measured != report.closure["measured"]:
            problems.append(
                f"the report's closure states {report.closure['measured']} measured")
        if measured != report.space["cells"]:
            problems.append(
                f"the report's space total states {report.space['cells']} cell(s)")
        if report.closure["refused"] != 0 or report.space["refused"] != 0:
            problems.append(
                f"the sentence says `no refusals` and the report states "
                f"{report.closure['refused']} refused in the closure and "
                f"{report.space['refused']} in the space")
        if problems:
            sites.append(Site(path, line, "run-counts", quote, "finding", "; ".join(problems)))
        else:
            sites.append(Site(path, line, "run-counts", quote, "agree",
                              f"the report's closure: {report.closure['possible']} possible = "
                              f"{report.closure['served']} served + "
                              f"{report.closure['refused']} refused, "
                              f"{report.closure['measured']} measured"))
    return sites


def check_batch_count(path: str, flat: str, index_map: list[int], original: str,
                      report: Report) -> list[Site]:
    sites = []
    block = report.classes.get(("fp64", "all-n"))
    if block is None:
        return sites
    for match in BATCH_COUNT.finditer(flat):
        line = line_of(original, index_map[match.start()])
        quote = f"{match.group('possible')} / {match.group('count')}"
        prefix = flat[max(0, match.start() - 300):match.start()]
        problems = []
        if "batch row" not in prefix:
            problems.append("the sentence does not name the batch row this check reads it as")
        if plain_int(match.group("possible")) != block.measured:
            problems.append(
                f"the report's fp64 all-n class measured {block.measured} combination(s)")
        if int(match.group("count")) != block.unplaced:
            problems.append(
                f"the report's fp64 all-n class has {block.unplaced} option(s) it could not place")
        if problems:
            sites.append(Site(path, line, "batch-count", quote, "finding", "; ".join(problems)))
        else:
            sites.append(Site(path, line, "batch-count", quote, "agree",
                              f"fp64 all-n measures {block.measured} combination(s) and has "
                              f"{block.unplaced} unplaced"))
    return sites


def check_facts(path: str, flat: str, index_map: list[int], original: str,
                report: Report) -> list[Site]:
    """The machine, backend and workload sentences, each against the report's own line."""
    sites = []

    for match in MACHINE_SENTENCE.finditer(flat):
        line = line_of(original, index_map[match.start()])
        quote = f"{match.group('cpus')} logical processors, {match.group('tier')} {match.group('state')}"
        problems = []
        if plain_int(match.group("cpus")) != report.cpus:
            problems.append(f"the report states {report.cpus} logical processors")
        if match.group("tier") != report.tier:
            problems.append(f"the report states the {report.tier} tier")
        if match.group("state") != report.tier_state:
            problems.append(f"the report states the tier is {report.tier_state}")
        sites.append(Site(path, line, "machine", quote,
                          "finding" if problems else "agree",
                          "; ".join(problems) or
                          f"the report's machine line: {report.cpus} logical processors, "
                          f"{report.tier} tier {report.tier_state}"))

    for match in BACKENDS_SENTENCE.finditer(flat):
        line = line_of(original, index_map[match.start()])
        quote = (f"{match.group('count')} backends, {match.group('roundings')} roundings")
        problems = []
        if number(match.group("count")) != len(report.backends):
            problems.append(f"the report names {len(report.backends)} arithmetic backend(s)")
        contracts = set(report.backends.values())
        expected = "two" if contracts == {False} else "one" if contracts == {True} else None
        if expected is None:
            problems.append("the report's backends do not agree on contraction, and the sentence "
                            "says each leaves the same rounding")
        elif match.group("roundings").lower() != expected:
            problems.append(
                f"the report's backends each read contracts={'yes' if contracts == {True} else 'no'}"
                f", which is {expected} rounding(s)")
        sites.append(Site(path, line, "backends", quote,
                          "finding" if problems else "agree",
                          "; ".join(problems) or
                          f"{len(report.backends)} backend(s), every one contracts=no and so leaves "
                          f"two roundings"))

    for match in WORKLOAD_SENTENCE.finditer(flat):
        line = line_of(original, index_map[match.start()])
        quote = (f"{match.group('arguments')} arguments over [{match.group('lo')}, "
                 f"{match.group('hi')}], orders {match.group('order_lo')}..{match.group('order_hi')}, "
                 f"nmax {match.group('nmax')}")
        report_workload = report.workload
        problems = []
        if plain_int(match.group("arguments")) != int(report_workload["arguments"]):
            problems.append(f"the report states {report_workload['arguments']} arguments")
        if not same_number(match.group("lo"), report_workload["lo"]):
            problems.append(f"the report's lower range end is {report_workload['lo']}")
        if not same_number(match.group("hi"), report_workload["hi"]):
            problems.append(f"the report's upper range end is {report_workload['hi']}")
        if match.group("order_lo") != report_workload["order_lo"]:
            problems.append(f"the report's lowest drawn order is {report_workload['order_lo']}")
        if match.group("order_hi") != report_workload["order_hi"]:
            problems.append(f"the report's highest drawn order is {report_workload['order_hi']}")
        if match.group("nmax") != report_workload["nmax"]:
            problems.append(f"the report states nmax {report_workload['nmax']}")
        sites.append(Site(path, line, "workload", quote,
                          "finding" if problems else "agree",
                          "; ".join(problems) or
                          f"the report's workload: {report_workload['arguments']} arguments on "
                          f"[{report_workload['lo']}, {report_workload['hi']}], orders "
                          f"{report_workload['order_lo']}..{report_workload['order_hi']}, "
                          f"nmax {report_workload['nmax']}"))
    return sites


def check_transcript(path: str, text: str, report: Report) -> list[Site]:
    """A transcript of RunOptionProbe's text: declared a different run, or a finding."""
    sites = []
    block = report.classes[("fp64", "all-orders")]
    for match in TRANSCRIPT.finditer(text):
        line = line_of(text, match.start())
        quote = (f"{match.group('measured')} options, {match.group('ranked')} ranked, "
                 f"{match.group('name')} at {match.group('ns')}, {match.group('unplaced')} unplaced")
        tail = text[match.end():match.end() + 900]
        declared = ("line counts are that build's" in tail
                    or "option space has changed size" in tail)
        if not declared:
            sites.append(Site(path, line, "example-transcript", quote, "finding",
                              "a RunOptionProbe transcript whose prose no longer says it is a "
                              "different, older run; a reader would take it for the committed "
                              "report"))
            continue
        sites.append(Site(
            path, line, "example-transcript", quote, "not-checked",
            f"`examples/07_option_probe.cpp`'s short protocol on the page's build, and the page "
            f"says its counts differ from the current build's. The report carries "
            f"{report.space['cells']} served cell(s) and a {block.measured}-option fp64 all-orders "
            f"class led by `{block.leader}` at {block.leader_ns}; the transcript's "
            f"{match.group('measured')}/{match.group('ranked')}/{match.group('ns')} and "
            f"`{match.group('name')}` are not cells of it. Not held."))
    return sites


def transcript_spans(text: str) -> list[tuple[int, int]]:
    """Each transcript block, from `measured ...` through the `recommended here:` line."""
    spans = []
    for match in TRANSCRIPT.finditer(text):
        end = match.end()
        recommended = text.find("recommended here:", end, end + 3000)
        if recommended >= 0:
            newline = text.find("\n", recommended)
            end = len(text) if newline < 0 else newline
        spans.append((match.start(), end))
    return spans


def check_names(path: str, text: str, report: Report, table_lines: set[int]) -> list[Site]:
    """Every option-name-like token the document prints must be a row of some class."""
    sites = []
    rows = report.rows()
    spans = transcript_spans(text)
    for match in OPTION_NAME.finditer(text):
        name = match.group(0)
        if name in report.backends:
            continue                                    # an arithmetic backend, not an option
        if any(start <= match.start() < end for start, end in spans):
            continue                                    # the declared different run
        line = line_of(text, match.start())
        if line in table_lines:
            continue                                    # already held as a class row
        if name in rows:
            sites.append(Site(path, line, "option-name", name, "agree",
                              "a row of the report's class tables"))
        else:
            sites.append(Site(path, line, "option-name", name, "finding",
                              "no row of any class in the report carries this name"))
    return sites


# --------------------------------------------------------------------------
# The declared set and the report reading
# --------------------------------------------------------------------------

def documents(root: pathlib.Path) -> list[pathlib.Path]:
    """Every declared document, refusing a set that is missing one it is about."""
    paths = [root / name for name in TOP_FILES]
    for dirname, pattern in SCAN_DIRS:
        base = root / dirname
        if not base.is_dir():
            raise Refusal(f"the declared set names {dirname}/ and {root} holds no such directory")
        paths.extend(sorted(base.glob(pattern)))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise Refusal("no such document: " + ", ".join(str(path) for path in missing))
    return paths


def inventory(sites: list[Site], root: pathlib.Path) -> list[Site]:
    """The sites the inventory requires, found or refused."""
    counts = Counter((site.path, site.kind) for site in sites)
    findings = []
    for path, kinds in EXPECTED.items():
        relative = path
        if not (root / path).is_file():
            findings.append(Site(path, 0, "inventory", path, "finding",
                                 "the inventory names this document and the tree holds no such "
                                 "file; the run is refused rather than made over what is left"))
            continue
        for kind, expected in kinds.items():
            found = counts[(relative, kind)]
            if found < expected:
                findings.append(Site(
                    path, 0, "inventory", kind, "finding",
                    f"the inventory requires {expected} site(s) of kind `{kind}` and this run "
                    f"found {found}. A checker that cannot find what it holds must not pass"))
    return findings


def run(root: pathlib.Path, report_path: pathlib.Path) -> tuple[Report, list[Site], list[Site]]:
    report = load_report(report_path)
    sites: list[Site] = []
    for path in documents(root):
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        flat, index_map = flatten(text)
        require_table = "class-row" in EXPECTED.get(relative, {})
        table = check_class_rows(relative, text, report, require_table)
        sites.extend(table)
        table_lines = {site.line for site in table}
        sites.extend(check_pair(relative, flat, index_map, text, report))
        sites.extend(check_documented(relative, flat, index_map, text, report))
        sites.extend(check_repeat(relative, flat, index_map, text, report))
        sites.extend(check_run_counts(relative, flat, index_map, text, report))
        sites.extend(check_batch_count(relative, flat, index_map, text, report))
        sites.extend(check_facts(relative, flat, index_map, text, report))
        sites.extend(check_transcript(relative, text, report))
        sites.extend(check_names(relative, text, report, table_lines))
    sites.extend(inventory(sites, root))
    return report, sites, []


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def report_output(report: Report, sites: list[Site], stream, show_agree: bool) -> int:
    findings = [site for site in sites if site.verdict == "finding"]
    agree = [site for site in sites if site.verdict == "agree"]
    not_checked = [site for site in sites if site.verdict == "not-checked"]

    order = {"finding": 0, "agree": 1, "not-checked": 2}
    sites = sorted(sites, key=lambda site: (order.get(site.verdict, 9), site.path, site.line,
                                            site.kind))

    stream.write("check_probe_quotes: the documents' quoted figures held to the option probe "
                 "report\n")
    stream.write(f"  report : {report.version} | inputs {report.inputs} | verdict "
                 f"{report.verdict}\n")
    stream.write(f"  classes: {len(report.classes)} read, "
                 f"{len([key for key in report.classes if key[0] == 'fp64'])} fp64 question "
                 f"shape(s)\n\n")

    if not sites:
        stream.write("check_probe_quotes: read the declared documents and found no site this "
                     "check holds. Prose that quotes nothing and prose whose quotes are all "
                     "correct report the same thing, and only the second is a clean set\n")
        return 1

    if findings:
        stream.write("findings\n")
        for site in findings:
            where = f"{site.path}:{site.line}" if site.line else site.path
            stream.write(f"  [FAIL] {where}  {site.kind}  {site.quote}\n")
            if site.detail:
                stream.write(f"         {site.detail}\n")
        stream.write("\n")

    if show_agree and agree:
        stream.write("agreements\n")
        for site in agree:
            stream.write(f"  [ok]   {site.path}:{site.line}  {site.kind}  {site.quote}\n")
            if site.detail:
                stream.write(f"         {site.detail}\n")
        stream.write("\n")

    stream.write("not checked (declared non-subjects)\n")
    for site in not_checked:
        stream.write(f"  --     {site.path}:{site.line}  {site.kind}  {site.quote}\n")
        if site.detail:
            stream.write(f"         {site.detail}\n")
    stream.write("  --     (declaration)  no declared site quotes a figure computed by combining "
                 "two report rows; the pair statements print two rows side by side and no ratio "
                 "of them, so there is no derived figure to hold\n")
    stream.write("  --     (declaration)  the `5.5e-14` a page other than getting-started states "
                 "is the lane's accuracy contract, not a cell of this report; it is held by the "
                 "contract's own checks and is not read as a probe quote here\n")
    stream.write("\n")

    stream.write(
        f"check_probe_quotes: {len(agree)} site(s) agree, {len(findings)} finding(s), "
        f"{len(not_checked)} declared not checked\n")
    if findings:
        stream.write("check_probe_quotes: FAIL: a quoted figure or name does not agree with the "
                     "report, a declared site is missing, or the report could not be read whole\n")
        return 1
    stream.write("check_probe_quotes: PASS: every quoted figure and name agrees with the report, "
                 "and every declared site was found\n")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Holds the documents' quoted option-probe figures to the recorded report.")
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero on a disagreement or a missing site")
    parser.add_argument("--root", type=pathlib.Path, default=REPO,
                        help="the tree the declared documents are read from")
    parser.add_argument("--report", type=pathlib.Path, default=REPORT,
                        help="the option probe report, relative to --root unless absolute")
    parser.add_argument("--show-agree", action="store_true",
                        help="print each site that agrees, not only the findings")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

    root = args.root
    report_path = args.report if args.report.is_absolute() else root / args.report

    try:
        report, sites, _ = run(root, report_path)
    except Refusal as error:
        print(f"check_probe_quotes: {error}", file=sys.stderr)
        print("check_probe_quotes: REFUSED: the check cannot find what it holds, so it does not "
              "report a clean set", file=sys.stderr)
        return 1

    return report_output(report, sites, sys.stdout, args.show_agree)


if __name__ == "__main__":
    raise SystemExit(main())
