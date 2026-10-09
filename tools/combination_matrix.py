#!/usr/bin/env python3
"""One matrix per target: every combination, whether it exists, whether it was measured, and
which of a class's combinations the seam in force resolves to.

The option probe's report is read as an account of the library, so a combination nothing measured
is read as a combination the library does not have. Four questions are answered of every class:

  1. CAN IT EXIST? A combination the axes offer. The axes and their members are the library's own,
     read from the probe's report, which reads them from ``BoysEvalSchemes``,
     ``BoysFitGranularities``, ``BoysPackAxes``, ``BoysDivisionForms`` and ``BoysRegionBExps``. The
     only fundamental exclusion is proved rather than asserted: a shape that evaluates ONE ORDER
     has one order to put in a vector lane, so the orders member of the packing axis does not exist
     on it. `--compile` instantiates, for each one-order shape, the combination with that member
     (must FAIL) and without it (must PASS).

  2. IS IT IMPLEMENTED? Host: the class's entry is declared and can be instantiated at the
     combination, which is what `--compile` measures, one translation unit per class carrying the
     whole cross. Device: an entry is a named kernel rather than a template, so a combination is
     implemented when the library's own option table (``kDeviceOptions``, which the library
     asserts carries one row per ``DeviceEntry``) places a row at those axes - there is no policy
     template on
     that side, so a class with no entry is NOT IMPLEMENTED, which is a defect and is named as one
     rather than as an absence.

  3. IS IT PROBED? The names the probe's report ranks in that class. A class's rows are the options
     measured as alternatives for its need, and the report states its own count of them; the names
     are counted here rather than trusted to the figure, so a report whose rows and whose figure
     disagree is caught rather than summarised.

  4. WHICH DEFAULT? The class has a row in the seam in force, and that row is one of the
     combinations the class was measured at.

THE DEVICE HAS TWO ROUTES AND THEY ARE SEPARATE FAMILIES
    A launched kernel (``BoysCuda::Xxx``) and an in-kernel device function (``BoysDeviceXxx``) are
    different work: one is a launch the caller makes, the other a body compiled into the caller's
    own kernel. A class on that side is therefore (device, precision, shape, ROUTE) and each route
    is complete or not on its own.

THE LIBRARY IS READ AT A REVISION, NOT FROM THE WORKING TREE
    A claim about what the library contains is a claim about a revision. The working tree can hold
    another writer's half-finished edit, and a report of that state is wrong about the library in a
    way no reader can see. Every header is read with ``git show <rev>:<path>``, and the revision is
    printed. `--compile` exports that revision to a scratch tree and compiles there, never the
    working tree, and it carries the compile definitions the library's own target publishes to its
    consumers - read from the build file, and a name the build file decides at configure time is
    named rather than guessed. A unit compiled without them is a unit compiled against a smaller
    surface than the report measured, which is how the half lane's classes once read as absent.

A RUN THAT CHECKS NOTHING EXITS 1
    Every reader here can fail silently: a name grammar that stopped matching, an axis key spelled
    differently from the lookup, a section the parser walked past. The count of things the run
    actually checked is therefore a printed number, and its zero is a failure and not a clean bill.

usage:
  combination_matrix.py --report <report.txt> [--root DIR] [--rev REV] [--compile]
                        [--device-report FILE] [--defaults FILE] [--show N]
"""

from __future__ import annotations

import argparse
import itertools
import pathlib
import re
import subprocess
import sys
import tempfile


# =============================================================================================
# Reading the library at a revision.
# =============================================================================================


class Reader:
    """The library read at one revision, or from a scratch tree already exported from one."""

    def __init__(self, root: pathlib.Path, rev: str):
        self.root = root
        self.rev = rev
        self._text: dict[str, str] = {}
        self.exported: pathlib.Path | None = None

    def text(self, relative: str) -> str:
        if relative in self._text:
            return self._text[relative]

        if self.exported is not None:
            path = self.exported / relative
            value = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        else:
            done = subprocess.run(
                ["git", "-C", str(self.root), "show", f"{self.rev}:{relative}"],
                capture_output=True, text=True, encoding="utf-8", errors="replace")
            value = done.stdout if done.returncode == 0 else ""

        self._text[relative] = value
        return value

    def export(self) -> pathlib.Path:
        """The revision written to a scratch tree, so a compile reads the revision, not the tree."""
        if self.exported is not None:
            return self.exported

        scratch = pathlib.Path(tempfile.mkdtemp(prefix="boys-rev-"))
        archived = subprocess.run(["git", "-C", str(self.root), "archive", self.rev],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        unpacked = subprocess.run(["tar", "-x", "-C", str(scratch)], input=archived.stdout)

        if archived.returncode != 0 or unpacked.returncode != 0:
            raise RuntimeError(f"git archive {self.rev} did not export: {archived.stderr!r}")

        self.exported = scratch
        return scratch

    def rev_id(self) -> str:
        done = subprocess.run(["git", "-C", str(self.root), "rev-parse", self.rev],
                              capture_output=True, text=True)
        return done.stdout.strip() or self.rev


def camel(member: str) -> str:
    """``all-n-at-orders`` -> ``AllNAtOrders``: the library's spelling of an axis member."""
    return "".join(word[:1].upper() + word[1:] for word in re.split(r"[-_ ]+", member) if word)


def read_function_table(reader: Reader, source: str, function: str) -> dict[str, str]:
    """One name-returning function of the probe, as its enumerator-to-spelling table.

    The probe's grammar is the authority on what a report's names mean, so the spellings are read
    out of the probe's own source rather than restated here. The enumerator that falls out of the
    switch unspelled is kept under the empty key, which is how ``LaneShapeName`` states the double
    lane's name.
    """
    text = reader.text(source)
    match = re.search(r"(?:const\s+char\s*\*|std::string)\s+" + re.escape(function)
                      + r"\s*\([^)]*\)[^{]*\{", text)

    if not match:
        return {}

    body = text[match.end():]
    end = body.find("\n}\n")
    body = body[:end if end >= 0 else len(body)]

    found = {name: spelling for name, spelling in
             re.findall(r"case\s+[A-Za-z_][A-Za-z0-9_]*::(k[A-Za-z0-9]+)\s*:\s*"
                        r"(?:\n\s*)*return\s+\"([^\"]*)\"\s*;", body)}

    # The member the switch falls out of unspelled: the function's own return below the switch is
    # what that member answers, and a reader that dropped it would read the double lane's cell as a
    # cell with no name. The case returns are taken out first, so the return left standing is the
    # one below the switch rather than the first case's.
    without_cases = re.sub(r"case\s+[A-Za-z_][A-Za-z0-9_]*::(k[A-Za-z0-9]+)\s*:\s*"
                           r"(?:\n\s*)*return\s+\"[^\"]*\"\s*;", "", body)
    default = re.search(r"return\s+\"([^\"]*)\"\s*;", without_cases)

    if default:
        found[""] = default.group(1)

    return found


# =============================================================================================
# The host report.
# =============================================================================================

AXIS_LINE = re.compile(r"^\s*axes:\s*(.*)$", re.M)
MEMBER_LIST = re.compile(r"\(([^)]*)\)")

# One class line of the report's accuracy-classes section:
#   "  fp64 all-orders       possible 144 = 144 served + 0 refused | 144 measured | fastest
#    uniform-pack-orders-horner-plain-reciprocal-accurate-fp64 at 88.66 ns/argument, documented
#    at 5.5e-14 | not ordered"
# The line states the class's own arithmetic - the cells it holds, the share this build serves and
# refuses, the share this run measured - and, where the class produced a figure, the fastest option
# with its cost, the figure it documents and whether the class was ordered. A class with no figure
# prints the counts and no fastest segment, so the segment is optional.
CLASS_LINE = re.compile(
    r"^\s{2}(?P<precision>[a-z0-9]+(?:-[a-z0-9]+)*)\s+(?P<shape>[a-z][a-z-]*?)\s+"
    r"possible\s+(?P<possible>\d+)\s*=\s*(?P<served>\d+)\s+served\s*\+\s*"
    r"(?P<refused>\d+)\s+refused\s*\|\s*(?P<measured>\d+)\s+measured"
    r"(?:\s*\|\s*fastest\s+(?P<fastest>[a-z0-9][a-z0-9-]*)\s+at\s+(?P<cost>[0-9.]+)\s*"
    r"ns/argument,\s*documented\s+at\s+(?P<bound>[0-9.eE+-]+)\s*\|\s*"
    r"(?P<order>not ordered|ordered))?", re.M)

FASTEST = re.compile(r"fastest\s+([a-z0-9][a-z0-9-]*)")

# The closure's class-by-class table and its own sums, which are what the class lines are held
# against: the table carries every class this build carries including the device lane's book, and
# the sums line states the space's total and this build's measured share of it.
CLASS_ROW = re.compile(
    r"^\s{4}(?P<precision>[a-z0-9]+(?:-[a-z0-9]+)*)\s+(?P<shape>[a-z][a-z-]*?)\s+"
    r"(?P<possible>\d+)\s+(?P<served>\d+)\s+(?P<measured>\d+)\s+(?P<refused>\d+)\s*$", re.M)
CLASS_SUMS = re.compile(
    r"the class-by-class sums:\s*(?P<possible>\d+)\s+possible\s*=\s*(?P<served>\d+)\s+served\s*\+\s*"
    r"(?P<refused>\d+)\s+refused.*?and\s+(?P<measured>\d+)\s+cell\(s\) measured", re.S)
DEVICE_LANE = re.compile(r"^\s{4}(?P<lane>[a-z0-9]+(?:-[a-z0-9]+)*) \(lane (?P=lane)\):", re.M)

# One ranked row of a class's book. The report prints several rows that tied on one line,
# separated by ", ", so the names are read from the whole line: a reader that takes the first name
# of a line and stops loses every row of every tie but one, which reads as a smaller library
# rather than as a parse that stopped early. The bound in parentheses is what tells a rank row
# from the per-cell figure book further down, whose lines carry a time and no bound.
RANKED_ROW = re.compile(r"([a-z0-9][a-z0-9-]*)\s+[0-9.]+ ns\s*\([0-9.eE+-]+\)")
RANK_INDENT = 13
SECTION_END = re.compile(r"^\S")


def axes_from(report: str) -> dict[str, list[str]]:
    """The axis members the report states, keyed by the name the report gives the axis."""
    lines = report.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip().startswith("axes:")), None)

    if start is None:
        return {}

    block = [lines[start].strip()[len("axes:"):]]

    for line in lines[start + 1:start + 6]:
        if "counted above" in line or not line.strip():
            break

        block.append(line.strip())

    axes: dict[str, list[str]] = {}

    for chunk in " ".join(block).split("|"):
        name = chunk.split("(")[0].strip().split()[1:] if "(" in chunk else []
        groups = MEMBER_LIST.findall(chunk)

        if not name or not groups:
            continue

        # The report spells the axes out ("division form", "packing axis/axes"); the lookup is by
        # the axis's own name, and a key that does not match it silently takes the default for
        # that axis - which collapses the axis and undercounts every class by its factor.
        key = " ".join(name).lower()
        for suffix in (" axis/axes", " axis", " form"):
            key = key.replace(suffix, "")

        axes[key.strip()] = [m.strip() for m in groups[-1].split(",")]

    return axes


def shapes_from(report: str) -> list[str]:
    """The question shapes the report lists, in the seam's own order."""
    lines = report.splitlines()
    start = next((i for i, line in enumerate(lines)
                  if "shapes here are" in line and "order" in line), None)

    if start is None:
        return []

    found: list[str] = []

    for line in lines[start + 1:start + 12]:
        match = re.match(r"^\s{4}([a-z][a-z-]*)\s{2,}\S", line)

        if match:
            found.append(match.group(1))
        elif found:
            break

    return found


def stated_counts_of(report: str) -> dict[tuple[str, str], dict]:
    """The figures each class line states, which the class's own book of rows must bear out.

    The report states a class's counts in one place and ranks that class's options in another, and
    the two are two readings of one fact. A tool that takes the stated figures alone cannot tell a
    report whose book lost its rows from a library that measured nothing, and a tool that counts
    the rows alone cannot tell the same two apart either; this is what lets both be checked. The
    line's own arithmetic - the cells it holds against the share this build serves and refuses, and
    the share this run measured - is stated with them, so a line whose parts do not add up is read
    as that rather than taken as a count.
    """
    found: dict[tuple[str, str], dict] = {}

    for match in CLASS_LINE.finditer(report):
        found[(match.group("precision"), match.group("shape"))] = {
            "possible": int(match.group("possible")),
            "served": int(match.group("served")),
            "refused": int(match.group("refused")),
            "measured": int(match.group("measured")),
            "fastest": match.group("fastest"),
            "cost": float(match.group("cost")) if match.group("cost") else None,
            "bound": match.group("bound"),
            "ordered": match.group("order"),
        }

    return found


def class_table(report: str) -> dict[tuple[str, str], dict]:
    """The closure's class-by-class table: every class this build carries, device book included."""
    found: dict[tuple[str, str], dict] = {}

    for match in CLASS_ROW.finditer(report):
        found[(match.group("precision"), match.group("shape"))] = {
            "possible": int(match.group("possible")),
            "served": int(match.group("served")),
            "measured": int(match.group("measured")),
            "refused": int(match.group("refused")),
        }

    return found


def class_sums(report: str) -> dict | None:
    """The closure's own totals over that table: the space's cells and this build's measured share."""
    match = CLASS_SUMS.search(report)

    if match is None:
        return None

    return {"possible": int(match.group("possible")), "served": int(match.group("served")),
            "refused": int(match.group("refused")), "measured": int(match.group("measured"))}


def device_lanes(report: str) -> set[str]:
    """The lanes the report counts apart from this build, from its own device-book statement."""
    return {match.group("lane") for match in DEVICE_LANE.finditer(report)}


# The resolution section's per-class block: a class, the refinement runs over its tied options, and
# the option the vote named. It is the only place the report states which combination a class
# settled on as opposed to which led the main book, and a class's winner is what a default row has
# to be able to carry.
VOTE_CLASS = re.compile(r"^\s{4}([a-z0-9]+) ([a-z-]+):\s+\d+ run\(s\)", re.M)
VOTE_NAMED = re.compile(r"^\s{6}vote:\s+([a-z0-9][a-z0-9-]*)\s+—", re.M)


def votes_by_class(report: str) -> dict[tuple[str, str], str]:
    """The option each class's refinement vote named, where the report took one.

    A class's block runs to the next class's, so the block is the text between one class heading and
    the next rather than up to the next line at any indentation: the vote line is indented under
    the heading, and a bound that takes the first line indented by four would take the vote line
    itself and read an empty block.
    """
    found: dict[tuple[str, str], str] = {}
    heading = list(VOTE_CLASS.finditer(report))

    for index, match in enumerate(heading):
        end = heading[index + 1].start() if index + 1 < len(heading) else len(report)
        named = VOTE_NAMED.search(report, match.end(), end)

        if named:
            found[(match.group(1), match.group(2))] = named.group(1)

    return found


def policy_cell(name: str) -> str:
    """A ranked name without the format it was measured in.

    Two names that differ only in their last segment are the same combination of the axes measured
    over two formats, and that is what a shared seam key has to be able to tell apart.
    """
    return re.sub(r"-(fp64|fp32|fp16|bf16)$", "", name)


def ranked_by_class(report: str) -> dict[tuple[str, str], list[str]]:
    """The option names the report ranks, per class, and the name each class leads with."""
    ranked: dict[tuple[str, str], list[str]] = {}
    current: tuple[str, str] | None = None

    for line in report.splitlines():
        header = CLASS_LINE.match(line)

        if header:
            current = (header.group("precision"), header.group("shape"))
            ranked.setdefault(current, [])
            continue

        if not line.strip():
            continue

        # A heading at column zero ends the classes: what follows is another section, and a class
        # seen after it would be a class the report never opened.
        if SECTION_END.match(line):
            current = None
            continue

        if current is None or (len(line) - len(line.lstrip())) < RANK_INDENT:
            continue

        ranked[current].extend(RANKED_ROW.findall(line))

    return ranked


def cell_name(precision: str, partition: str, pack: str, route: str, scheme: str, division: str,
              exp: str) -> str:
    """The name ``CellName`` gives one combination: the probe's grammar, segment for segment.

    The partition's segment is the shipped one's own word, the packing axis is named only where it
    is the orders member, the route only where it is the rational one, the scheme only where it is
    Horner, and the division form and the region-B exponential only where they differ from the
    seam's defaults. The last two segments are what stop two cells colliding on one name: a cell
    at the plain or exact form would otherwise print the default's spelling, and a cell seeded with
    the other exponential would print the fast one's.
    """
    partition_word = {"shipped": "batch"}.get(partition, partition)
    name = partition_word

    if pack == "orders":
        name += "-pack-orders"

    if route == "rational-minimax":
        name += "-rational"

    if scheme == "horner":
        name += "-horner"

    if division != "refined-reciprocal":
        name += "-" + division

    if exp != "fast":
        name += "-" + exp

    return name + "-" + precision


# =============================================================================================
# The seam in force, and the device table.
# =============================================================================================

SEAM_CALL = re.compile(r"X\(([^)]*)\)")
PRECISION_ENUM = re.compile(r"^\s*(k[A-Za-z0-9]+)\s*(?:=[^,]+)?,\s*///<\s*(.*)$", re.M)


def join_continuations(text: str) -> str:
    """The seam's rows as one line each: a row is written across as many lines as its width needs."""
    return re.sub(r"\\\s*\n\s*", " ", text)


# =============================================================================================
# The names a class's cells are printed under, read from the probe and from the surface.
# =============================================================================================

# One row of ``CarriedOwnCellNames``: the two names the report has carried for a class's own cell
# and for its sorted twin, either of which the probe writes as nullptr where the grammar's own
# name is the one the report carries.
CARRIED_GATE = re.compile(r"shape\s*!=\s*OptionProbeShape::(k[A-Za-z0-9]+)")
CARRIED_ROW = re.compile(r"case\s+OptionPrecision::(k[A-Za-z0-9]+)\s*:\s*return\s*\{\s*"
                         r"(nullptr|\"[^\"]*\")\s*,\s*(nullptr|\"[^\"]*\")\s*\}\s*;", re.S)
SORTED_SEGMENT = re.compile(r"\bkSortedCellSegment\s*=\s*\"([^\"]*)\"")
SORTED_OVERLOAD = re.compile(r"\b(Boys[A-Za-z0-9_]+)\s*\(([^;{]*)\)")


def carried_own_cells(reader: Reader) -> tuple[str | None, dict[str, tuple[str | None, str | None]]]:
    """The class names the report carried before its grammar did, read out of the probe's table.

    ``CarriedOwnCellNames`` states, for one shape alone, the name a class's own cell has been
    printed under and the one its sorted twin carries; every other class prints the grammar's own
    name. The shape the table is gated on is read with it, because a reader that took the table for
    every shape would put the double lane's carried names on four questions that never had them.
    The segment a sorted twin takes where no name was carried is read here too, so that the spelling
    the report prints is the probe's own.
    """
    text = reader.text("src/boys_probe.cpp")
    match = re.search(r"constexpr\s+CarriedNames\s+CarriedOwnCellNames\s*\([^)]*\)[^{]*\{", text)

    if not match:
        return None, {}

    body = text[match.end():]
    end = body.find("\n}\n")
    body = body[:end if end >= 0 else len(body)]

    gate = CARRIED_GATE.search(body)
    found: dict[str, tuple[str | None, str | None]] = {}

    for row in CARRIED_ROW.finditer(body):
        found[row.group(1)] = tuple(None if cell == "nullptr" else cell.strip('"')
                                    for cell in (row.group(2), row.group(3)))

    return (gate.group(1) if gate else None), found


def sorted_segment(reader: Reader) -> str:
    """The segment the probe appends to a cell's name for the call that skips the sort."""
    match = SORTED_SEGMENT.search(reader.text("src/boys_probe.cpp"))

    return match.group(1) if match else ""


def sorted_overload_entries(reader: Reader) -> set[str]:
    """The entries whose declaration carries the ``BoysSortedArgs`` overload, from the surface.

    The probe asks this question of the compiler, with a ``requires`` expression over the entry
    itself; a reader without a compiler asks the same question of the declaration, which is the
    thing that expression resolves against. A class whose entry declares the overload carries a
    second cell per combination - the same combination reached by the call that skips the sort -
    and a tool that counted one cell there would read the run's book as double what its grammar
    states.
    """
    found: set[str] = set()

    for header in ("boys.hpp", "boys_span.hpp", "boys_half.hpp"):
        text = join_continuations(reader.text(f"include/boys/{header}"))

        for match in SORTED_OVERLOAD.finditer(text):
            if "BoysSortedArgs" in match.group(2):
                found.add(match.group(1))

    return found


def seam_rows(text: str) -> dict[tuple[str, str, str], dict[str, str]]:
    """The default rows of a seam file: (device, precision, shape) -> the combination's cells.

    A row is ``X(device, precision, shape, route, scheme, budget, packing, granularity, division,
    exponential, basis, record)``. The cells are the library's own spellings; which axis member
    each names is resolved by the caller, which is the only place that holds the report's naming of
    the axes. The last two cells are the row's provenance - what the combination is and which run
    it came from - and are not axes of it, so they are not read here.
    """
    found: dict[tuple[str, str, str], dict[str, str]] = {}
    keys = ("route", "scheme", "budget", "packing", "granularity", "division", "exponential")

    for call in SEAM_CALL.findall(join_continuations(text)):
        cells = [cell.strip() for cell in call.split(",")]

        if len(cells) != 3 + len(keys) + 2 or not cells[0].startswith("k"):
            continue

        device = cells[0][1:].lower()
        found[(device, cells[1], cells[2])] = dict(zip(keys, cells[3:3 + len(keys)]))

    return found


def pairings(reader: Reader, source: str, functions: list[str]) -> dict[str, str]:
    """The library's own pairing of an axis enumerator with the name it publishes.

    Each axis is a table of rows carrying the enumerator beside its name, or a switch answering the
    name for an enumerator, and the two are the same fact stated twice. The report names its axis
    members with the published name, so this is what turns a member back into the enumerator a
    seam row writes - the shipped partition is ``FitGranularity::kCoarsest`` and pairs with
    "shipped" here, which no rule of spelling would reach.

    A name is claimed by the first enumerator that pairs with it. Two enumerators publishing one
    name would make a member ambiguous, which is a thing to refuse rather than to resolve.
    """
    text = reader.text(source)
    found: dict[str, str] = {}
    ambiguous: set[str] = set()

    for function in functions:
        match = re.search(r"\b" + re.escape(function) + r"\s*\([^)]*\)[^{]*\{", text)

        if not match:
            continue

        body = text[match.end():]
        end = body.find("\n}\n")
        body = body[:end if end >= 0 else len(body)]

        pairs = re.findall(r"([A-Za-z_][A-Za-z0-9_]*::(?:k[A-Za-z0-9]+))\s*,\s*\n?\s*\"([^\"]*)\"",
                           body)
        pairs += re.findall(r"case\s+[A-Za-z_][A-Za-z0-9_]*::(k[A-Za-z0-9]+)\s*:\s*(?:\n\s*)*"
                            r"return\s+\"([^\"]*)\"\s*;", body)

        for enumerator, name in pairs:
            enumerator = enumerator.split("::")[-1]

            if name in found and found[name] != enumerator:
                ambiguous.add(name)
            else:
                found[name] = enumerator

    for name in ambiguous:
        found.pop(name, None)

    return found


def member_map(axis_members: list[str], seam_spellings: set[str],
               published: dict[str, str]) -> tuple[dict[str, str], set[str]] | None:
    """Which enumerator each of an axis's members names, and which members were spelled, not read.

    The pairing is the library's where the library publishes one. A member the library's tables do
    not name is paired by spelling - ``plain-reciprocal`` -> ``kPlainReciprocal`` - and returned in
    the second set so that a reader that leaned on that rule can say so.

    A spelling the seam uses that no member reaches is refused: a row written at it could not be
    read back, and a row silently compared against the wrong member is worse than no answer.
    """
    found: dict[str, str] = {}
    spelled: set[str] = set()

    for member in axis_members:
        enumerator = published.get(member)

        if enumerator is None:
            enumerator = "k" + camel(member)
            spelled.add(member)

        found[member] = enumerator

    unread = sorted(seam_spellings - set(found.values()))

    if unread:
        return None

    return found, spelled


def enum_members(text: str, name: str) -> dict[str, str]:
    """One enum's members, in the order the enum states them, each with its own documentation."""
    start = text.find(f"enum class {name}")

    if start < 0:
        return {}

    end = text.find("};", start)

    return {member: doc for member, doc in DEVICE_MEMBER.findall(text[start:end])
            if member != "kCount"}


def kebab(member: str) -> str:
    """``AllOrders`` -> ``all-orders``: the class label of an enumerator named ``kAllOrders``."""
    return re.sub(r"(?<!^)(?=[A-Z])", "-", member).lower()


def device_entries(rows: str, options: str) -> tuple[dict[str, tuple[str, str, str, str]],
                                                     list[str], int]:
    """The device option space, read from the library's own table of it.

    ``kDeviceOptions`` (src/boys_cuda.cpp) carries one row per ``DeviceEntry``, in enumerator order,
    and each row states the group a caller reaches the entry by, the precision it computes in, the
    shape it answers with and the question that shape answers. Those four fields are what this
    reads, and not the enumerator's spelling: the spelling is a name, and the group a row belongs to
    is a fact the library states.

    The question is read beside the shape because the two are not one field: a shape delivered to a
    sink and the same shape written to an array are one question, and the seam's device rows are
    keyed by the question (``SeamShapeCell`` in the device probe). A reader that took the shape for
    the class key would look for a class the seam does not carry.

    The row's group is a member of ``DeviceOptionGroup``, whose own documentation tells the two
    routes apart - one is queued by this library and the other is inlined into the caller's kernel -
    so which member means which route is read rather than written here.

    :returns the entries, keyed by enumerator, as (route family, precision member, shape member,
    question member); the rows whose enumerator comment names a route family their row's group does
    not, which is two statements of one fact disagreeing and so a defect rather than a preference;
    and the number of enumerators the enumeration declares, which the caller checks the table
    against.
    """
    groups = enum_members(options, "DeviceOptionGroup")
    queued = [member for member, doc in groups.items() if QUEUED_DOC.search(doc)]
    inlined = [member for member, doc in groups.items() if INLINED_DOC.search(doc)]

    if len(queued) != 1 or len(inlined) != 1 or queued == inlined:
        return {}, [], 0

    routes = {queued[0]: "launched", inlined[0]: "in-kernel"}
    comments = enum_members(options, "DeviceEntry")
    found: dict[str, tuple[str, str, str, str]] = {}
    disagreements: list[str] = []

    for entry, group, precision, shape, question in DEVICE_ROW.findall(rows):
        route = routes.get(group)

        if route is None:
            continue

        stated = comments.get(entry, "")
        named = ("launched" if stated.startswith("BoysCuda::") else
                 "in-kernel" if stated.startswith("BoysDevice") else None)

        if named is not None and named != route:
            disagreements.append(
                f"{entry}: the enumeration's comment names a {named} entry and the option table's "
                f"row places it in {route} ({group})")

        found[entry] = (route, precision, shape, question)

    return found, disagreements, len(comments)


def carrier_of(precision: str, carriers: dict[str, str]) -> str | None:
    """Which ``Precision`` member documents carrying this format, where the obvious one does not.

    One member carries two formats, so the class the enumeration has no key of its own for is
    reached through the key whose own documentation names the format. The match is on the format's
    two spellings, and it is only consulted after the key of the class's own name has been tried.
    """
    long_form = {"fp64": ("double", "fp64"), "fp32": ("float", "fp32"),
                 "fp16": ("fp16", "half"), "bf16": ("bfloat16", "bf16")}.get(precision)

    if not long_form:
        return None

    pattern = re.compile(r"\b(" + "|".join(re.escape(w) for w in long_form) + r")\b")

    return next((name for name, doc in carriers.items() if pattern.search(doc)), None)


def resolve(row: dict[str, str], maps: dict[str, dict[str, str]]) -> dict[str, str]:
    """A seam row read back into the report's own names for the axis members.

    A row names its cells in the library's spelling and the report names the same members in its
    own; the pairing between the two is ``maps``, worked out once against the seam. A cell no
    member pairs with empties the result, so a row this tool cannot read is reported as unread
    rather than silently compared against the wrong member.
    """
    found: dict[str, str] = {}

    for axis, cell in (("route", row["route"]), ("scheme", row["scheme"]),
                       ("partition", row["granularity"]), ("packing", row["packing"]),
                       ("division", row["division"]), ("exponential", row["exponential"])):
        spelling = cell.split("::")[-1]
        member = next((m for m, s in maps.get(axis, {}).items() if s == spelling), None)

        if member is None:
            return {}

        found[axis] = member

    return found


def precision_carriers(text: str) -> dict[str, str]:
    """The formats each ``Precision`` member carries, per the enumeration's own documentation.

    One member of that enum carries two formats - ``kFp16`` is the fp16 and bfloat16 entries - and
    a seam keyed by the member therefore cannot carry a separate row for each. This reads the
    library's own statement of that rather than restating it.
    """
    start = text.find("enum class Precision")

    if start < 0:
        return {}

    end = text.find("};", start)
    found: dict[str, str] = {}

    for name, doc in PRECISION_ENUM.findall(text[start:end]):
        found[name] = doc

    return found


# =============================================================================================
# The host surface.
# =============================================================================================

HOST_ENTRY = {
    "single": "BoysSingle",
    "all-orders": "BoysAllOrders",
    "fixed-n": "BoysFixedN",
    "all-n": "BoysAllN",
    "all-n-at-orders": "BoysAllNAtOrders",
}
SUFFIX = {"fp64": "", "fp32": "F32", "fp16": "F16", "bf16": "Bf16"}
SCALAR = {"fp64": "double", "fp32": "float", "fp16": "boys::F16", "bf16": "boys::Bf16"}
BUDGET = {"fp64": "kFloat", "fp32": "kFloat", "fp16": "kFp16", "bf16": "kFp16"}
INIT = {"fp64": "1.5", "fp32": "1.5f", "fp16": "static_cast<boys::F16>(1.5f)",
        "bf16": "static_cast<boys::Bf16>(1.5f)"}
DECLARATION = re.compile(r"\b(Boys[A-Za-z0-9_]+)\s*\(")

# The library target's own statement of what a consumer's translation unit needs defined, e.g.
#   target_compile_definitions(boys PUBLIC BoysFp16=1)
# ``boys.hpp`` gates the whole half-precision surface on ``BoysFp16`` and states no default for it,
# so a unit compiled without the definition is a unit compiled against a smaller library than the
# one the report measured. Which definitions those are is read from the build file and never
# written here: a definition this tool made up would be the tool measuring its own idea of the
# surface.
DEFINITIONS = re.compile(r"target_compile_definitions\(\s*boys\s([^)]*)\)", re.S)
DEFINITION_KEYWORD = re.compile(r"\b(PRIVATE|PUBLIC|INTERFACE)\b")
CONFIGURED = re.compile(r"\b(?:option|set|unset)\(\s*([A-Za-z_]\w*)")

SHAPE_ENUM = re.compile(r"^\s*(k[A-Za-z0-9]+)\s*(?:=\s*\d+)?\s*,?\s*///<\s*(.*)$", re.M)
ONE_ORDER = re.compile(r"\bone order\b", re.I)

# The device option table, its rows, and the two group members' own documentation. The table is
# written in src/boys_cuda.cpp and asserted there to carry one row per enumerator in enumerator
# order; this reader holds its count against the enumeration's own for that reason.
DEVICE_ROW = re.compile(
    r"\{\s*DeviceEntry::(k\w+)\s*,\s*\"[^\"]*\"\s*,\s*DeviceOptionGroup::(k\w+)\s*,\s*"
    r"DeviceOptionPrecision::(k\w+)\s*,\s*DeviceOptionShape::(k\w+)\s*,\s*"
    r"DeviceOptionQuestion::(k\w+)\s*,", re.S)
DEVICE_MEMBER = re.compile(r"^\s*(k[A-Za-z0-9]+)\s*(?:=\s*\d+)?\s*,?\s*///<\s*(.*)$", re.M)
QUEUED_DOC = re.compile(r"\bqueued by this library\b", re.I)
INLINED_DOC = re.compile(r"\binlined into the caller's kernel\b", re.I)

# The device lane's precision cell in the build's seam: the option table's precision member
# carries the lane, and the seam spells the cell after it. The pairing is reported as a spelling
# one, because that is what it is - the library publishes no table joining the two.
SEAM_DEVICE_PRECISION = "Device"


def one_order_shapes(reader: Reader) -> set[str]:
    """The ``Shape`` enumerators whose own documentation names one order."""
    text = reader.text("include/boys/boys.hpp")
    start = text.find("enum class Shape")

    if start < 0:
        return set()

    end = text.find("};", start)

    return {name for name, doc in SHAPE_ENUM.findall(text[start:end]) if ONE_ORDER.search(doc)}


def host_entry(precision: str, shape: str) -> str:
    return HOST_ENTRY[shape] + SUFFIX[precision]


# The library's own spelling of each axis member, filled in by main() from the pairing it works
# out against the seam's spellings. Most members camel-case into their enumerator; the shipped
# partition does not (it is FitGranularity::kCoarsest), which is the case this indirection exists
# for. Nothing is instantiated before main() has made the pairing, and a member the pairing did not
# reach falls back to the camel-case spelling so that a hole is a compile error rather than a
# silently different axis member.
ENUMS: dict[str, dict[str, str]] = {}


def enum_of(axis: str, member: str) -> str:
    return ENUMS.get(axis, {}).get(member, "k" + camel(member))


def instantiation(precision: str, shape: str, combination: dict[str, str]) -> str:
    """One line instantiating a class's entry at one combination."""
    policy = (
        "boys::EvalPolicy<"
        f"boys::FitRoute::{enum_of('route', combination['route'])}, "
        f"boys::EvalScheme::{enum_of('scheme', combination['scheme'])}, "
        f"boys::BoysBudget::{BUDGET[precision]}, "
        f"boys::PackAxis::{enum_of('packing', combination['packing'])}, "
        f"boys::FitGranularity::{enum_of('partition', combination['partition'])}, "
        f"boys::DivisionForm::{enum_of('division', combination['division'])}, "
        f"boys::RegionBExp::{enum_of('exponential', combination['exponential'])}>")

    entry = host_entry(precision, shape)

    if shape == "single":
        return f"    (void)boys::{entry}<{policy}>(4, x);"
    if shape == "all-orders":
        return f"    boys::{entry}<{policy}>(4, x, out);"
    if shape == "fixed-n":
        return f"    boys::{entry}<{policy}>(4, xs, out, 2, 1);"
    if shape == "all-n":
        return f"    boys::{entry}<{policy}>(4, xs, out, 2);"

    return f"    boys::{entry}<{policy}>(ns, xs, out, 2);"


def unit_for(precision: str, shape: str, combinations: list[dict[str, str]]) -> str:
    lines = [
        "#include <boys/boys.hpp>",
        "void Instantiate() {",
        f"    [[maybe_unused]] {SCALAR[precision]} x = {INIT[precision]};",
        f"    [[maybe_unused]] {SCALAR[precision]} xs[8] = {{}};",
        f"    [[maybe_unused]] {SCALAR[precision]} out[64] = {{}};",
        "    [[maybe_unused]] int ns[8] = {};",
        "    (void)xs; (void)out; (void)ns;",
    ]

    for combination in combinations:
        lines.append(instantiation(precision, shape, combination))

    lines.append("}")
    return "\n".join(lines) + "\n"


def build_definitions(tree: pathlib.Path) -> tuple[list[str], list[str]]:
    """The compile definitions a consumer of the headers inherits, and the names left out.

    ``boys.hpp`` gates the half-precision surface on ``BoysFp16``, which the build states as a
    PUBLIC compile definition and the header states no default for: a unit compiled without it is a
    unit compiled against a smaller library than the one the report measured. Which definitions
    those are is read from the build file at the revision and never written here.

    A name whose value this tool cannot read is not guessed. The build file states some of them
    under a configure-time condition - a cache entry, an option, a hash of another file - and one
    of those branches is not the other; carrying either would be this tool asserting a build fact
    it did not read. Those names are returned beside the carried ones so the run can say which
    definitions its compile did not carry and why.
    """
    path = pathlib.Path(tree) / "CMakeLists.txt"
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    values: dict[str, set[str]] = {}
    unreadable: dict[str, str] = {}

    for call in DEFINITIONS.finditer(text):
        parts = DEFINITION_KEYWORD.split(call.group(1))

        for index in range(1, len(parts) - 1, 2):
            if parts[index] not in ("PUBLIC", "INTERFACE"):
                continue

            for token in re.split(r"[\s;]+", parts[index + 1]):
                name = token.split("=", 1)[0]

                if not re.fullmatch(r"[A-Za-z_]\w*", name) or "$" in token or '"' in token:
                    unreadable.setdefault(name, "its value is worked out at configure time")
                    continue

                values.setdefault(name, set()).add(token)

    # A name the build file sets, options or clears governs the definitions spelled under it: a
    # definition the file states inside the branch a cache entry decides is a definition the
    # shipped configure does not carry, and this tool read no configure.
    for name in sorted(values):
        governing = [c for c in sorted({m.group(1) for m in CONFIGURED.finditer(text)})
                     if name == c or name.startswith(c + "_")]

        if governing:
            unreadable.setdefault(
                name, f"the build file decides it at configure time, under {governing[0]}")

    carried: list[str] = []
    skipped: list[str] = []

    for name in sorted(values):
        if name in unreadable:
            skipped.append(f"{name} ({unreadable[name]})")
        elif len(values[name]) > 1:
            skipped.append(f"{name} (the build file states "
                           f"{len(values[name])} different values, one per configure branch)")
        else:
            carried.append(sorted(values[name])[0])

    return carried, skipped


def compile_unit(tree: pathlib.Path, source: str, definitions: list[str]) -> bool:
    """Whether one scratch translation unit compiles against the exported revision.

    The definitions are the ones the library's own target publishes, because a unit compiled
    without them is compiled against a different surface than the report measured - the half lane
    that ``boys.hpp`` gates behind ``BoysFp16`` is exactly that case, and a tool that omitted the
    definition would report eight classes as ones the library cannot be instantiated at when the
    truth is that the tool asked the compiler the wrong question.

    The compiler's own message is printed when the unit fails: a compile that fails silently is a
    finding no reader can check.
    """
    with tempfile.TemporaryDirectory() as scratch:
        unit = pathlib.Path(scratch) / "unit.cpp"
        unit.write_text(source, encoding="utf-8")
        log = pathlib.Path(scratch) / "log.txt"
        flags = " ".join(f"/D{definition}" for definition in definitions)
        script = pathlib.Path(scratch) / "run.bat"
        script.write_text(
            "@echo off\n"
            'call "C:\\Program Files\\Microsoft Visual Studio\\18\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat" >nul 2>&1\n'
            f'cd /d "{tree}"\n'
            f'cl /nologo /std:c++latest /EHsc /Zs {flags} /I include "{unit}" > "{log}" 2>&1\n',
            encoding="utf-8")

        try:
            done = subprocess.run(["cmd.exe", "/c", str(script)], capture_output=True, timeout=900)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("      the compiler did not answer before this unit's deadline")
            return False

        if done.returncode == 0:
            return True

        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""

        for line in [l for l in text.splitlines() if " error " in l or "fatal error" in l][:4]:
            print(f"      {line.strip()[:160]}")

        return False


# =============================================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="the option probe's report")
    parser.add_argument("--root", default=".")
    parser.add_argument("--rev", default="HEAD",
                        help="the revision the library is read at (never the working tree)")
    parser.add_argument("--compile", action="store_true",
                        help="establish what is implemented by instantiating (slow, and the "
                             "control the rest of the numbers rest on)")
    parser.add_argument("--device-report", default=None,
                        help="the device probe's report, if one has been run")
    parser.add_argument("--defaults", default=None,
                        help="the seam in force: the emitted file, or the committed one")
    parser.add_argument("--show", type=int, default=6, help="names to print per finding")
    arguments = parser.parse_args()

    root = pathlib.Path(arguments.root).resolve()
    reader = Reader(root, arguments.rev)
    report = pathlib.Path(arguments.report).read_text(encoding="utf-8", errors="replace")

    print(f"library read at {arguments.rev} ({reader.rev_id()}), never from the working tree")

    axes = axes_from(report)
    shapes = shapes_from(report)
    ranked = ranked_by_class(report)
    stated_counts = stated_counts_of(report)
    closure = class_table(report)
    sums = class_sums(report)
    apart = device_lanes(report)
    votes_by_class_of = votes_by_class(report)

    if not closure or sums is None:
        print("combination_matrix: the report states no class-by-class table or no sums over it, "
              "so the counts its class lines carry have nothing to be held against and this run "
              "checked far less than it reads as having checked")
        return 1

    if len(axes) != 6:
        print(f"combination_matrix: the report states {len(axes)} axis/axes where the policy has "
              "six, so no combination can be generated from it")
        return 1

    if not shapes:
        print("combination_matrix: the report states no question shape, so no class can be named "
              "from it")
        return 1

    if not ranked:
        print("combination_matrix: the report ranks no option in any class, so this run measured "
              "nothing and its exit status is a failure rather than a clean bill")
        return 1

    keys = list(axes)

    # The probe's own grammar, read from the probe's source.
    probe_precision = read_function_table(reader, "src/boys_probe.cpp", "PrecisionName")
    lane_name = read_function_table(reader, "src/boys_probe.cpp", "LaneShapeName")

    if not probe_precision or not lane_name:
        print("combination_matrix: PrecisionName and LaneShapeName were not both read out of "
              "src/boys_probe.cpp, so no name this tool builds can be checked against the "
              "probe's own spelling")
        return 1

    by_precision = {value: key for key, value in probe_precision.items() if key}

    # The rest of the naming: the shape the report has carried names for, those names, the segment
    # a sorted twin takes, and the entries that declare the sorted overload at all.
    shape_words = read_function_table(reader, "src/boys_probe.cpp", "OptionProbeShapeName")
    carried_shape_enum, carried = carried_own_cells(reader)
    sorted_suffix = sorted_segment(reader)
    sorted_entries = sorted_overload_entries(reader)

    if not shape_words or not sorted_suffix or not sorted_entries:
        print("combination_matrix: the probe's shape names, its sorted-cell segment and the "
              "surface's sorted-arguments overloads were not all read - a name this tool builds "
              "would then be its own spelling rather than the probe's, so the reader is wrong, "
              "not the library")
        return 1

    carried_shape = shape_words.get(carried_shape_enum) if carried_shape_enum else None
    all_orders_shape = shape_words.get("kAllOrders")

    if (carried and carried_shape is None) or all_orders_shape is None:
        print("combination_matrix: the probe's own shape table does not spell a shape this tool "
              "names classes by - the reader is wrong, not the library")
        return 1

    surface = set()
    for header in ("boys.hpp", "boys_span.hpp", "boys_half.hpp"):
        surface.update(DECLARATION.findall(reader.text(f"include/boys/{header}")))

    if not surface:
        print("combination_matrix: the headers declare no entry, so the host surface was read as "
              "empty - the reader is wrong, not the library")
        return 1

    one_order = one_order_shapes(reader)

    if not one_order:
        print("combination_matrix: no Shape enumerator's documentation names one order, so the "
              "exclusion this matrix rests on was not read - the reader is wrong, not the library")
        return 1

    one_order_tokens = {name[1:].lower() for name in one_order}

    # The seam in force: the committed one, read at the revision, and an emitted file only where a
    # caller names it. A scratch seam that happens to lie in the working tree is not this tree's
    # default and must not be read as one - it is the output of a run, and a run's own seam is a
    # different question from the one this check asks. Reading it here once turned a device class
    # the committed seam carries into "the seam in force carries no row for it", which is a finding
    # about the file that lay on disk and not about the library.
    if arguments.defaults:
        seam_text = pathlib.Path(arguments.defaults).read_text(encoding="utf-8", errors="replace")
        seam_origin = str(pathlib.Path(arguments.defaults))
    else:
        seam_text = reader.text("include/boys/boys_build_defaults.hpp")
        seam_origin = f"{arguments.rev}:include/boys/boys_build_defaults.hpp"

    seam = seam_rows(seam_text)

    if not seam:
        print(f"combination_matrix: the seam at {seam_origin} carries no default row, so no class "
              "can be held to one")
        return 1

    print("axes, read from the library through the report:")
    for key, members in axes.items():
        print(f"  {key:<20} {len(members)}  {', '.join(members)}")
    print(f"question shapes, read from the report: {', '.join(shapes)}")
    print(f"seam in force: {seam_origin} ({len(seam)} row(s))")

    defects: list[str] = []

    # ---------------------------------------------------------------------------------------
    # The host classes.
    # ---------------------------------------------------------------------------------------

    # A class is one precision and one question shape. The shape list is the report's; the
    # precisions are the ones its class lines carry, checked against the spellings the probe's own
    # PrecisionName answers, so a class the report names in a word the probe does not spell is a
    # reading this tool refuses rather than a class it quietly drops.
    precisions = list(dict.fromkeys(
        precision for precision, _ in ranked if precision in by_precision))

    if not precisions:
        print("combination_matrix: no class line in the report names a precision the probe's own "
              "PrecisionName spells, so the class list was not read")
        return 1

    # A seam row writes each cell qualified (`FitRoute::kChebyshev`); the member that names is the
    # enumerator, which is the last segment.
    def spellings(cell: str) -> set[str]:
        return {row[cell].split("::")[-1] for row in seam.values()}

    published = pairings(reader, "src/boys.cpp",
                         ["BoysFitRoutes", "BoysEvalSchemes", "BoysPackAxes", "BoysRegionBExps",
                          "BoysDivisionForms", "BoysFitGranularities", "EvalSchemeName",
                          "PackAxisName", "GranularityName", "DivisionFormName", "RegionBExpName"])

    if not published:
        print("combination_matrix: src/boys.cpp published no enumerator-to-name pair, so an axis "
              "member cannot be turned into the enumerator a seam row writes - the reader is "
              "wrong, not the library")
        return 1

    tables = {"route": member_map(axes.get("route", ["chebyshev", "rational-minimax"]),
                                  spellings("route"), published),
              "scheme": member_map(axes.get("scheme", []), spellings("scheme"), published),
              "partition": member_map(axes.get("partition", []), spellings("granularity"),
                                      published),
              "packing": member_map(axes.get("packing", []), spellings("packing"), published),
              "division": member_map(axes.get("division", []), spellings("division"), published),
              "exponential": member_map(axes.get("exponential", []), spellings("exponential"),
                                        published)}

    for axis, table in tables.items():
        if table is None:
            print(f"combination_matrix: the seam's spellings of the {axis} axis do not pair with "
                  "the members the report states, so a default row cannot be read back into a "
                  "combination - the reader is wrong, not the library")
            return 1

    maps = {axis: table[0] for axis, table in tables.items()}
    spelled = sorted({member for _, table in tables.items() for member in table[1]})

    if spelled:
        print(f"axis members the library's tables do not publish a name for, paired by spelling: "
              f"{', '.join(spelled)}")

    ENUMS.update(maps)

    carriers = precision_carriers(reader.text("include/boys/boys.hpp"))

    if not carriers:
        print("combination_matrix: the Precision enumeration was read as carrying no format, so no "
              "class's key can be found - the reader is wrong, not the library")
        return 1

    # The one fundamental exclusion, and its proof where the compiler is asked for it. It is asked
    # after the axis members have been paired, because the unit it compiles must name the library's
    # own enumerators: a control that compiled a spelling the library does not have would fail for
    # the spelling and pass for nothing.
    tree = None
    definitions: list[str] = []

    if arguments.compile:
        tree = reader.export()
        definitions, not_carried = build_definitions(tree)

        if not definitions:
            print("\ncombination_matrix: the revision's build file states no PUBLIC compile "
                  "definition for the library target that this tool can read, so a unit compiled "
                  "here would be compiled against a surface it cannot claim is the report's - the "
                  "reader is wrong, not the library")
            return 1

        print(f"\ncompiling the revision exported to {tree}, carrying the build's own "
              f"definition(s): {', '.join(definitions)}")

        if not_carried:
            print("  not carried, and named rather than guessed: " + "; ".join(not_carried))

        for shape in [s for s in shapes if s in HOST_ENTRY
                      and s.replace("-", "").lower() in one_order_tokens]:
            at = dict(zip(keys, ["chebyshev", "horner", "shipped", "orders",
                                 "refined-reciprocal", "fast"]))
            off = dict(at, packing="arguments")

            if compile_unit(tree, unit_for("fp32", shape, [at]), definitions) or \
                    not compile_unit(tree, unit_for("fp32", shape, [off]), definitions):
                print(f"  [control FAILED] {shape}: the orders member is not refused as the "
                      "exclusion assumes, so no count below is admissible")
                return 1

            print(f"  [control] {shape}: the orders member is refused by the compiler, the "
                  "arguments member is accepted")

    print("\n  class                    possible  implemented   probed  missing  note            "
          "default")
    total_possible = 0
    total_probed = 0
    total_stated = 0
    total_served = 0
    total_refused = 0
    total_measured = 0
    rows = []
    shape_rows: list[tuple[str, str]] = []

    for precision, shape in [(p, s) for p in precisions for s in shapes]:
        members = [axes[k] for k in keys]

        if shape.replace("-", "").lower() in one_order_tokens:
            members[keys.index("packing")] = [m for m in members[keys.index("packing")]
                                              if m != "orders"]

        combinations = [dict(zip(keys, combination))
                        for combination in itertools.product(*members)]

        # The name each of the class's cells is printed under, in the probe's own rules. The
        # grammar's name is the probe's, segment for segment; every class but the all-orders one
        # carries its shape's word in front of it, because one combination of the six run-time
        # axes carries five questions and a name that did not say which one a row answers would be
        # one name on five cells. The class's own cell is the combination whose grammar name
        # carries no axis segment at all - the shipped partition on the arguments axis at the
        # shipped route and scheme - and it prints under the name the report has carried for that
        # call where it has one, which is a fact of the probe's table and not of this tool's
        # spelling.
        own_grammar = "batch-" + precision
        prefix = "" if shape == all_orders_shape else shape + "-"
        lane_key = by_precision.get(precision, "")

        if shape == all_orders_shape:
            # The all-orders class's own cell is the lane's own name for the call - `batch-fp64`
            # on the double lane, and the format-boundary names on the half lanes, whose entries
            # are the single-precision engine with a store on either side.
            own_name = lane_name.get(lane_key, lane_name.get("", ""))
        else:
            own_name = shape + "-" + precision

        carved = carried.get(lane_key) if shape == carried_shape else None

        if carved and carved[0]:
            own_name = carved[0]

        if not own_name:
            print(f"combination_matrix: the probe's own tables name no cell for the class "
                  f"{precision} {shape}, so its book cannot be read against a name this tool "
                  "builds - the reader is wrong, not the library")
            return 1

        own_sorted = carved[1] if carved and carved[1] else own_name + sorted_suffix

        # Whether the class carries a second cell per combination: the entry's declaration is what
        # the probe's own trait resolves against, so the declaration is what this reads.
        second = host_entry(precision, shape) in sorted_entries
        names = set()

        def printed_name(cell: dict[str, str]) -> str:
            """One combination of this class, under the name the probe prints it under."""
            grammar = cell_name(precision, cell["partition"], cell["packing"], cell["route"],
                                cell["scheme"], cell["division"], cell["exponential"])

            return own_name if grammar == own_grammar else prefix + grammar

        for combination in combinations:
            printed = printed_name(combination)
            names.add(printed)

            if second:
                names.add(own_sorted if printed == own_name else printed + sorted_suffix)

        # A class with no entry has no combination anything can be instantiated at, which is not
        # the same as a class whose combinations exist and went unmeasured: the first is a hole in
        # the library and the second a hole in the run. The first is named and counts zero, so the
        # total this tool prints is a statement about instantiating and not about coverage.
        implemented = host_entry(precision, shape) in surface
        compiled_here: bool | None = None

        if not implemented:
            defects.append(
                f"{precision} {shape}: the headers declare no {host_entry(precision, shape)}, so "
                f"the class cannot be instantiated at any of its {len(names)} combination(s) - a "
                "defect, not an absence")
        elif tree is not None:
            # One translation unit carrying the class's whole cross. A declaration is the surface
            # saying a class exists; instantiating it at every combination is the compiler saying
            # each of those combinations is one it can be built at, which is the claim this column
            # makes and the reason it is a count rather than a yes.
            print(f"  compiling {precision} {shape}: {len(combinations)} combination(s)",
                  flush=True)
            compiled_here = compile_unit(tree, unit_for(precision, shape, combinations),
                                         definitions)

            if not compiled_here:
                implemented = False
                defects.append(
                    f"{precision} {shape}: the class's entry does not compile at one or more of "
                    f"the {len(names)} combination(s) its axes offer, so those combinations are "
                    "not ones the library can be instantiated at")

        # The class's rows, as a set: the report prints a row once, and a class that ranked one
        # name twice would be a report whose book double-counts.
        measured = sorted(set(ranked.get((precision, shape), [])) - {""})

        known = [name for name in measured if name in names]
        unknown = [name for name in measured if name not in names]

        # What the class line states, and what the closure's own table states for the same class.
        # The line's counts are read as an arithmetic that has to hold - the cells it holds against
        # the share the build serves and refuses, and the share this run measured - and the closure's
        # row is a second reading of the same three counts, so a class whose own line and whose row
        # disagree is named rather than averaged.
        stated = stated_counts.get((precision, shape))
        row_stated = closure.get((precision, shape))

        possible = len(names) if implemented else 0

        if stated is None:
            defects.append(f"{precision} {shape}: the report states no class line for this class, "
                           "so nothing about it is asserted to hold and this row is the tool's own "
                           "reading alone")
        else:
            if stated["served"] + stated["refused"] != stated["possible"]:
                defects.append(
                    f"{precision} {shape}: the class line states {stated['possible']} possible but "
                    f"{stated['served']} served + {stated['refused']} refused - the line's own "
                    "arithmetic does not hold")

            if stated["measured"] and stated["measured"] != stated["served"]:
                defects.append(
                    f"{precision} {shape}: the class line states {stated['measured']} measured of "
                    f"{stated['served']} served, so the share this run measured is not the share "
                    "this build serves")

            if stated["fastest"] and stated["fastest"] not in measured:
                defects.append(
                    f"{precision} {shape}: the class line names `{stated['fastest']}` as the "
                    "class's fastest option and the class's own book ranks no such name, so the "
                    "line names a figure the book below it does not carry")

            if stated["measured"] != len(known) and stated["measured"] != len(measured):
                defects.append(
                    f"{precision} {shape}: the class line states {stated['measured']} measured and "
                    f"its own book ranks {len(measured)} name(s) - one of the two readings is "
                    "wrong and neither figure is usable")

            if implemented and stated["possible"] != len(names) and stated["measured"] != 0:
                defects.append(
                    f"{precision} {shape}: the class line states {stated['possible']} cell(s) and "
                    f"the axes the report states give this class {len(names)} "
                    f"(one per combination, and {'a second' if second else 'no second'} where the "
                    "entry declares the sorted-arguments overload)")

        if row_stated is not None and stated is not None:
            for figure in ("possible", "served", "refused"):
                if row_stated[figure] != stated[figure]:
                    defects.append(
                        f"{precision} {shape}: the class line states {stated[figure]} {figure} and "
                        f"the closure's class-by-class table states {row_stated[figure]}, so the "
                        "report carries two figures for one class")

            if row_stated["measured"] != stated["measured"]:
                defects.append(
                    f"{precision} {shape}: the class line states {stated['measured']} measured and "
                    f"the closure's class-by-class table states {row_stated['measured']}, so the "
                    "report carries two figures for one class")

        # The default leg. A class has a default of its own when the seam in force carries a row
        # keyed at it AND that row names a combination the class was measured at; a row naming a
        # combination nothing measured is a default chosen off no figure. A class whose format the
        # seam's key does not name is reached through whichever key's documentation does name it,
        # which on this seam is the one key that carries two formats.
        seam_key = "k" + camel(precision)
        key_owner = seam_key if seam_key in carriers else carrier_of(precision, carriers)
        row = seam.get(("host", key_owner, "k" + camel(shape))) if key_owner else None

        # A class the library has no entry for is a class no caller can ask at any combination, so
        # the seam's row for it is not a default it can be handed: that defect is the missing
        # entry, reported above, and not a second one here.
        if key_owner is None:
            default = "no key"
            defects.append(f"{precision} {shape}: no key of the seam's Precision enumeration "
                           f"documents the {precision} format, so this class has no default to "
                           "read")
        elif row is None:
            default = "no row"

            if implemented:
                defects.append(f"{precision} {shape}: the seam carries no default row for this "
                               "class, and the class has a policy-templated entry - a class an "
                               "entry reaches is a class the table must carry a row for")
        else:
            cell = resolve(row, maps)

            if not cell:
                default = "row?"

                if implemented:
                    defects.append(f"{precision} {shape}: the seam's row names a combination this "
                                   "tool cannot read back into the report's axis members, so "
                                   "whether it is a measured combination is unproven")
            elif printed_name(cell) in set(known) or unknown:
                # Either the seam's combination is one the class ranked, or the class ranked a
                # name its own grammar does not produce, which this tool cannot place on any
                # combination of the class and names rather than assumes. The name the seam's row
                # is held to is the one the class's own cells are printed under - the grammar's
                # name with its shape's word in front of it, or the carried name on the own cell -
                # because a name built for the comparison alone would be a second grammar, and a
                # class whose book ranks the seam's combination under the probe's own spelling
                # would read as a class whose default nothing measured.
                default = "row" if key_owner == seam_key else f"row via {key_owner}"

                if key_owner != seam_key:
                    # The key carries two formats. Whether that is a hole depends on whether the
                    # two classes settled on one combination: where they did, one row states both
                    # defaults and nothing is lost; where they did not, the row can be the default
                    # of one of them only, and the other gets a combination it did not choose.
                    others = [p for p in precisions
                              if p != precision and "k" + camel(p) == key_owner]
                    theirs = votes_by_class_of.get((others[0], shape)) if others else None
                    mine = votes_by_class_of.get((precision, shape))

                    other = others[0] if others else "the other half"

                    if mine and theirs and policy_cell(mine) == policy_cell(theirs):
                        # One combination for both, so the one row states both defaults and
                        # nothing is lost to the shared key.
                        default = f"row via {key_owner}"

                    elif mine and theirs:
                        default = f"row via {key_owner} (not this format's)"
                        defects.append(
                            f"{precision} {shape}: the seam carries one row for both half "
                            f"formats, keyed `{key_owner}`, and the two settled on different "
                            f"combinations: {other}'s refinement vote named `{theirs}` and "
                            f"{precision}'s named `{mine}`. One row cannot be both defaults, and "
                            f"this class gets {cell_name(precision, cell['partition'], cell['packing'], cell['route'], cell['scheme'], cell['division'], cell['exponential'])!r}, "
                            "which it did not choose. A default of its own needs a format key the "
                            "enumeration does not have.")

                    elif implemented:
                        # Only one of the two took a vote, so whether the two settled together is
                        # not something this run states. The row is this class's shape-row
                        # measurement at worst, which is a figure; that it is this format's winner
                        # is what is unproven.
                        default = f"row via {key_owner}"
                        defects.append(
                            f"{precision} {shape}: the class's default is the row keyed "
                            f"`{key_owner}`, which the half lane shares with {other}: only one of "
                            "the two classes took a refinement vote, so whether one row states "
                            "both formats' defaults is unproven by this run.")
            elif implemented:
                default = "row, unmeasured"
                defects.append(
                    f"{precision} {shape}: the seam's row names a combination this class did not "
                    "measure, so the class's default is a combination with no figure behind it")
            else:
                default = "not served"

        # Unmeasured is what the class offers less what the report ranks in it. A class's rows can
        # outnumber its combinations - the half lane's shape row is a name and not a combination -
        # which is a fact about the probe's book and not a negative amount of work, so the count
        # floors at zero and the over-count is reported where the name is read.
        rows.append((precision, shape, possible, len(measured), max(0, possible - len(known)),
                     len(known), unknown, stated,
                     "NO ENTRY" if not implemented else
                     f"{len(names)}/{len(names)}" if compiled_here else
                     "declared" if compiled_here is None else "REFUSED",
                     default))
        total_possible += possible
        total_probed += len(measured)
        total_served += stated["served"] if stated else 0
        total_refused += stated["refused"] if stated else 0
        total_stated += stated["possible"] if stated else 0
        total_measured += stated["measured"] if stated else 0
        shape_rows.extend((f"{precision} {shape}", name) for name in unknown)

    print("\n  the class line, and this tool's own reading of the same class. The first four "
          "columns are what\n  the report states; `cells` is what the axes it states give the "
          "class, and `missing` is what that\n  leaves unranked in the class's own book:")
    print("  class                    possible  served  refused  measured     cells  probed  "
          "missing  note            default")

    for (precision, shape, possible, probed, missing, known, unknown, stated, implemented,
         default) in rows:
        label = f"{precision} {shape}"
        note = f"{len(unknown)} unknown name(s)" if unknown else ""
        figures = (stated["possible"], stated["served"], stated["refused"], stated["measured"]) \
            if stated else (0, 0, 0, 0)
        print(f"  {label:<24} {figures[0]:>8} {figures[1]:>7} {figures[2]:>8} {figures[3]:>9}  "
              f"{possible:>9}  {probed:>6}  {missing:>7}  {note:<15} {default}")

    host_unmeasured = sum(row[4] for row in rows)
    print(f"\n  class(es) the library serves:                    {len(rows)}")
    print(f"  cell(s) their class lines state:                 {total_stated} = {total_served} "
          f"served + {total_refused} refused, {total_measured} measured")
    print(f"  cell(s) the axes the report states give them:    {total_possible}")
    print(f"  cell(s) their books rank:                        {total_probed}")
    print(f"host total — cell(s) a class can be instantiated at and this run did not rank: "
          f"{host_unmeasured}")

    # The space's own total, which the host classes' lines are not the whole of: the closure
    # carries the device lane's book beside this build and counts it apart, so the total the class
    # lines add up to and the total the space states are two figures that have to differ by
    # exactly that book and by nothing else.
    print(f"\n  the closure's own sums: {sums['possible']} possible = {sums['served']} served + "
          f"{sums['refused']} refused, {sums['measured']} measured")

    if not apart:
        defects.append(
            "the space: the report states no book it counts apart from this build, so the class "
            "lines' total cannot be squared with the space's own total - the reader is wrong, "
            "not the report")

    apart_rows = {key: row for key, row in closure.items() if key[0] in apart}
    apart_totals = {figure: sum(row[figure] for row in apart_rows.values())
                    for figure in ("possible", "served", "measured", "refused")}

    if apart:
        print(f"  counted apart, from the closure's own table ({len(apart_rows)} row(s) over "
              f"{', '.join(sorted(apart))}): {apart_totals['possible']} possible = "
              f"{apart_totals['served']} served + {apart_totals['refused']} refused, "
              f"{apart_totals['measured']} measured")

    # The class lines are the classes this build carries; the closure's table is the space. The
    # two totals have to differ by exactly the book the report counts apart and by nothing else,
    # so a class line missing from the table, or a row the lines do not add up to, is a number
    # here rather than a rounding of the space.
    for figure, total in (("possible", total_stated), ("served", total_served),
                          ("refused", total_refused), ("measured", total_measured)):
        if total + apart_totals[figure] != sums[figure]:
            defects.append(
                f"the space: the class lines state {total} {figure} cell(s) and the book the "
                f"report counts apart adds {apart_totals[figure]}, while the closure's own sums "
                f"state {sums[figure]} - the space's total is not the two together")

    if total_possible != total_stated:
        defects.append(
            f"the space: the axes the report states give the classes {total_possible} cell(s) and "
            f"their own class lines state {total_stated}, so the axes and the lines are two "
            "readings of one space that disagree")

    unserved = [f"{precision} {shape}"
                for precision, shape, *_, implemented, _ in rows
                if implemented == "NO ENTRY"]

    if unserved:
        print(f"\n{len(unserved)} class(es) the library has no entry for at all, so they hold no "
              f"combination to measure: {', '.join(unserved)}")

    if shape_rows:
        print(f"\n{len(shape_rows)} row(s) the report ranks that name no combination of their "
              "class: each is appended at the class's own default policy, so it measures the "
              "class's default cell under the lane's own name for the question rather than under "
              "the axes' name -")
        for label, name in shape_rows:
            print(f"    {label:<24} {name}")

    # A run that checked no class has found nothing, which is not the same as finding nothing
    # wrong. This is the one answer that must never read as success: it is what a surface read
    # that failed to parse looks like from the outside.
    if total_possible == 0:
        print("\nno class was checked: the surface or the axes were read as empty, so this run "
              "states nothing about the library and its exit status is a failure rather than a "
              "clean bill")
        return 1

    # ---------------------------------------------------------------------------------------
    # The device classes.
    # ---------------------------------------------------------------------------------------

    options_text = reader.text("include/boys/boys_cuda_options.hpp")
    table, route_disagreements, declared_entries = device_entries(
        reader.text("src/boys_cuda.cpp"), options_text)

    if not table:
        print("\ncombination_matrix: the device option space was read as empty, so no device class "
              "was checked - the reader is wrong, not the library")
        return 1

    if len(table) != declared_entries:
        print(f"\ncombination_matrix: the option table states {len(table)} row(s) where "
              f"DeviceEntry declares {declared_entries} enumerator(s), so rows were read past - "
              "the reader is wrong, not the library")
        return 1

    # The class is (precision, question, ROUTE) and not (precision, shape, ROUTE): a shape the
    # library delivers to a sink and the same shape written to an array are one question, which is
    # what the option table's own question cell states and what the seam's device rows are keyed by.
    # The shape-to-question folding is read from the table, so it is the library's that is used and
    # not a rule written here; where the table folds nothing, the two readings are the same one.
    print("\nthe device half — a class is (precision, question, ROUTE), and each route is complete "
          "or not on its own")
    counts: dict[tuple[str, str, str], int] = {}

    for _, (route, precision, shape, question) in table.items():
        counts[(precision, question, route)] = counts.get((precision, question, route), 0) + 1

    # The cell the seam spells for a question, which is the question's own shape: read from the
    # device probe's table rather than assumed, because a class the table carries and the seam has
    # no key for is exactly what a spelling taken for granted would hide.
    seam_cells = read_function_table(reader, "src/boys_cuda_probe.cpp", "SeamShapeCell")

    if not seam_cells:
        print("combination_matrix: the device probe's own question-to-seam-cell table was not "
              "read, so no device class can be held to a default row - the reader is wrong, not "
              "the library")
        return 1

    shapes_of = {shape for _, _, shape, _ in table.values()}
    questions_of = {question for _, _, _, question in table.values()}
    folded = {question: sorted({shape for _, _, shape, q in table.values() if q == question})
              for question in questions_of}
    print(f"  the table's own question cell places its {len(shapes_of)} shape(s) in "
          f"{len(questions_of)} question(s): "
          + "; ".join(f"{kebab(q[1:])} = {' + '.join(kebab(s[1:]) for s in folded[q])}"
                      for q in sorted(questions_of)))

    question_order = list(enum_members(options_text, "DeviceOptionQuestion"))
    device_questions = sorted(questions_of, key=lambda q: question_order.index(q)
                              if q in question_order else len(question_order))
    device_precisions = sorted({p for p, _, _ in counts})
    device_routes = sorted({route for _, _, route in counts})

    # The seam's device rows are keyed by the precision cell it spells, and the pairing between
    # that cell and the option table's precision member is a spelling this tool made: state it, and
    # refuse a cell no member reaches, so that a cell read past cannot look like a class with no
    # default row.
    seam_precisions = {key[1] for key in seam if key[0] == "device"}
    unreached = sorted(seam_precisions - {p + SEAM_DEVICE_PRECISION for p in device_precisions})

    if unreached:
        print(f"  the seam's device row(s) are keyed {', '.join(unreached)}, which names no lane of "
              "the option table - the reader is wrong, not the library")
        return 1

    print(f"  the seam keys a device row with {', '.join(sorted(seam_precisions))}: the lane's "
          "precision member under the seam's own spelling, which is the pairing the default column "
          "reads and the one thing here this tool pairs rather than reads")
    print("  class                              entries  default row")
    device_defects: list[str] = list(route_disagreements)

    if len(device_routes) != 2:
        print(f"  the device table states {len(device_routes)} route family(ies) where the surface "
              "has two - the reader is wrong, not the library")
        return 1

    for question in device_questions:
        for precision in device_precisions:
            for route in device_routes:
                count = counts.get((precision, question, route), 0)
                seam_row = seam.get(("device", precision + SEAM_DEVICE_PRECISION,
                                     seam_cells.get(question, "")))
                has_default = "row" if seam_row else "no row"

                if count == 0:
                    other = "in-kernel" if route == "launched" else "launched"
                    note = "NOT IMPLEMENTED — no kernel of this class in the DeviceEntry table"
                    device_defects.append(
                        f"device {kebab(precision[1:])} {kebab(question[1:])} {route}: the library "
                        "names no kernel of this class, and the other route names "
                        f"{counts.get((precision, question, other), 0)} — a route that is complete "
                        "on its own would carry both")
                else:
                    note = ""

                    if seam_row is None:
                        # A class the table has entries for and the seam has no row for is a class
                        # a chooser cannot ask for its default: the entries are reachable by hand
                        # and the class's own answer is not reachable at all.
                        note = "NO DEFAULT ROW — entries exist and no seam row names this class"
                        device_defects.append(
                            f"device {kebab(precision[1:])} {kebab(question[1:])} {route}: the "
                            f"table names {count} kernel(s) of this class and the seam in force "
                            "carries no row for it, so nothing states which of them the class "
                            "resolves to")

                label = (f"device {kebab(precision[1:]):<6} {kebab(question[1:]):<12} {route:<10}")
                print(f"  {label} {count:>8}  {has_default:<11} {note}")

    device_total = sum(counts.values())
    print(f"\n  device entries read from DeviceEntry: {device_total}")
    print(f"  device classes with entries:         {len(counts)}")
    print(f"  device classes with no entry:        "
          f"{len(device_questions) * len(device_precisions) * len(device_routes) - len(counts)}")

    if arguments.device_report:
        print(f"\ndevice report: {arguments.device_report}")
    else:
        print("\nno device report exists yet: nothing has ranked the device half, so its PROBED "
              "leg is unmeasured. The IMPLEMENTED and DEFAULT legs above are read from the "
              "headers and the seam, which is what this half rests on until a "
              "`boys-device-probe` run exists.")

    # ---------------------------------------------------------------------------------------

    for defect in defects + device_defects:
        print(f"\nDEFECT: {defect}")

    print(f"\nhost total — combination(s) a class can be instantiated at and this run did not "
          f"measure: {host_unmeasured}")

    # A device defect is a class of the space the library does not serve or does not answer for,
    # which is the same failure the host defects are: the run does not pass while one stands.
    if defects or device_defects:
        return 1

    if host_unmeasured == 0:
        print("\nevery combination a class can be instantiated at is a combination the report "
              "ranks")
        return 0

    print(f"\n{host_unmeasured} combination(s) can be instantiated at and nothing measured. Each "
          "is a cell of a class whose entry is policy-templated, so it is a combination the "
          "library serves and the probe did not ask for.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
