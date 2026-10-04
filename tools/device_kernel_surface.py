#!/usr/bin/env python3
r"""Enumerate the device half's option space per class and say which combinations it serves.

The condition applied here is the owner's, on the device half: for every class - a device precision
lane, an output shape and a route - take the axis combinations the library's own surfaces can
express, and say of each whether an entry implements it. An unimplemented combination is a
**defect** unless the library's own words prove it fundamentally impossible; an implemented
combination nothing probed is a bug, and that half is the probe's book rather than this tool's.

Where the enumeration comes from. Every fact is read from the library at one revision, never listed
beside it:

  * the entries and their axes - `include/boys/boys_cuda_options.hpp`, the `DeviceEntry`
    enumeration with `DeviceEntryAxesOf` and `DevicePartitionOf`;
  * the option table - `src/boys_cuda.cpp`, `kDeviceOptions`, which states each row's group,
    precision, shape, question, axis, region-B exponential and lane;
  * the two surfaces - `include/boys/boys_cuda.hpp` (launched) and
    `include/boys/boys_cuda_device.hpp` (in-kernel), read for the function each entry names, for
    the division form each function takes, and for the policy templates either of them declares;
  * the axis member sets - the library's own enumerations, `boys/accuracy.hpp` and
    `DevicePacking` in the options header;
  * the kernels' bodies - `src/boys_cuda.cu`, read for the tag-parameterised body each launched
    kernel hands its lane to, which is what makes a missing row a bounded job or not.

The space per class is the product of the axes the shape admits over the members the library's own
enumerations declare, and each applicability rule is printed with the sentence it is read from. The
space is deliberately larger than the option table: the table is what is built, and this tool
exists to count what the space admits and the table does not carry.

The three states of a combination, no fourth:

  * `ok`         - an entry implements it: the enumerator, the surface function and the option
    table's own printed name;
  * `IMPOSSIBLE` - no entry implements it and the library's own words prove none can: the sentence
    is quoted with its file and line, and the quotation is checked to occur verbatim before it is
    printed;
  * `DEFECT`     - no entry implements it and no impossibility is stated. "Not written yet" is a
    defect and not an explanation, so a missing combination falls here by default. Where another
    lane or the other route implements the same combination the tool names that counterpart, which
    is what makes this one a bounded job; where the library names the family as owed or as this
    build's scope, its sentence is printed beside the verdict and the verdict stays a defect.

Nothing here is compiled and no probe is run: the tool reads text at a revision and enumerates over
it. It resolves HEAD once and reads every file at that revision, because a working tree may have a
writer in it and HEAD itself moves while other lanes commit.

Usage:

    python3 tools/device_kernel_surface.py                    # the whole enumeration
    python3 tools/device_kernel_surface.py --summary          # class totals only
    python3 tools/device_kernel_surface.py --missing          # the work item, combination by combination
    python3 tools/device_kernel_surface.py --check            # exit 1 on an empty run or a defect
    python3 tools/device_kernel_surface.py --source worktree  # read the working tree instead of HEAD
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import re
import subprocess
import sys
from typing import NamedTuple

REPO = pathlib.Path(__file__).resolve().parent.parent

OPTIONS = "include/boys/boys_cuda_options.hpp"
LAUNCHED = "include/boys/boys_cuda.hpp"
INKERNEL = "include/boys/boys_cuda_device.hpp"
ROWS = "src/boys_cuda.cpp"
KERNELS = "src/boys_cuda.cu"
ENUMS = "include/boys/accuracy.hpp"
PROBE = "include/boys/boys_probe.hpp"
ENTRIES = "src/boys_cuda_probe_entries.hpp"

SOURCES = (OPTIONS, LAUNCHED, INKERNEL, ROWS, KERNELS, ENUMS, PROBE, ENTRIES)

# The sentinels an enumeration may end with, on the headers' own words: "one past the last" and
# the value a switch answers for an enumerator it has not been taught. Neither is a member.
SENTINELS = ("kCount", "kUnstated")

LAUNCHED_GROUP = "kLaunched"
DEVICE_GROUP = "kDeviceCallable"


class Source(NamedTuple):
    """One file of the library, at one revision."""

    path: str
    revision: str
    text: str


def read_sources(source: str) -> tuple[str, dict[str, Source]]:
    """The library's text, and the revision it was read at."""
    if source == "worktree":
        revision = "worktree"
        texts = {path: (REPO / path).read_text(encoding="utf-8", errors="replace")
                 for path in SOURCES}
    else:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                                  text=True, encoding="utf-8", errors="replace",
                                  check=True).stdout.strip()
        texts = {}

        for path in SOURCES:
            # The library's headers are UTF-8 and the host locale is not: the encoding is named
            # rather than inferred, so a run cannot fail on a byte outside the locale's charset.
            # Every file is read at the revision resolved once above, never at HEAD again: other
            # lanes commit to this repository while this tool runs, and a revision that moved
            # mid-read would put two revisions' text under one heading.
            texts[path] = subprocess.run(["git", "show", f"{revision}:{path}"], cwd=REPO,
                                         capture_output=True, text=True, encoding="utf-8",
                                         errors="replace", check=True).stdout

    return revision, {path: Source(path, revision, text) for path, text in texts.items()}


def quote_line(text: str, quote: str) -> int:
    """The 1-based line `quote` begins on, or 0 when the source does not carry it.

    A sentence in these headers is prose broken across `///` lines, so a quotation of one is not a
    substring of the file: the search is made over the comments' own text, with the markers and the
    line breaks collapsed to single spaces, and each character of that text is carried back to the
    line it came from. A quotation this returns 0 for is never printed: the caller refuses instead,
    so a sentence that has drifted from the source cannot be parried as though it had not.
    """
    characters: list[str] = []
    origins: list[int] = []

    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith("///"):
            body = stripped[3:].strip()
        elif stripped.startswith("//"):
            body = stripped[2:].strip()
        else:
            body = stripped

        if not body:
            continue

        if characters:
            characters.append(" ")
            origins.append(number)

        characters.extend(body)
        origins.extend([number] * len(body))

    index = "".join(characters).find(" ".join(quote.split()))

    return origins[index] if index >= 0 else 0


def enumerate_members(text: str, enum: str) -> list[str]:
    """The enumerators `enum` declares, minus the sentinel it ends with."""
    match = re.search(rf"enum class {enum}\s*:\s*[\w:]+\s*\{{(.*?)\n\}};", text, re.DOTALL)

    if match is None:
        raise SystemExit(f"device_kernel_surface: no enum class {enum} in the source read. A tool "
                         f"that can pass by reading nothing is not a tool")

    members = re.findall(r"^\s*(k[A-Za-z0-9_]+)\s*(?:=[^,]*)?,", match.group(1), re.MULTILINE)

    if not members:
        raise SystemExit(f"device_kernel_surface: enum class {enum} declares no enumerator")

    if members[-1] in SENTINELS:
        members = members[:-1]

    return members


def function_body(text: str, signature: str) -> str:
    """The body of the function whose declaration contains `signature`."""
    start = text.find(signature)

    if start < 0:
        raise SystemExit(f"device_kernel_surface: the source names no `{signature}`")

    brace = text.find("{", start)

    if brace < 0:
        raise SystemExit(f"device_kernel_surface: `{signature}` has no body")

    depth = 0

    for index in range(brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1

            if depth == 0:
                return text[brace:index + 1]

    raise SystemExit(f"device_kernel_surface: `{signature}` has no closing brace")


def switch_arms(body: str) -> list[tuple[list[str], list[str]]]:
    """A switch's arms as (the cases, the cells of the return they state).

    A return belongs to every `case` label since the previous return: that is how the header shares
    one statement between several enumerators. Both shapes the header uses are read - a braced list
    of cells (`return {A, B, C, D};`) and a bare enumerator (`return FitGranularity::kUniform;`) -
    because a parser that read only the first would walk one of the two switches as if it stated
    nothing.
    """
    arms: list[tuple[list[str], list[str]]] = []
    cases: list[str] = []

    for match in re.finditer(r"case DeviceEntry::(\w+):|return\s+(.*?);", body, re.DOTALL):
        if match.group(1) is not None:
            cases.append(match.group(1))
            continue

        content = match.group(2).strip()

        if content.startswith("{") and content.endswith("}"):
            content = content[1:-1]

        cells = [cell.strip() for cell in content.split(",")]
        arms.append((list(cases), cells))
        cases = []

    return arms


class Entry(NamedTuple):
    """One enumerator of DeviceEntry, with every coordinate the library states for it."""

    name: str
    order: int
    route: str
    scheme: str
    packing: str
    partition: str
    division: str
    surface: str
    comment: str


def read_entries(options: Source) -> list[Entry]:
    """Every entry, with the axes the library's own switches state for it."""
    text = options.text
    enum_match = re.search(r"enum class DeviceEntry\s*:\s*[\w:]+\s*\{(.*?)\n\};", text, re.DOTALL)

    if enum_match is None:
        raise SystemExit("device_kernel_surface: the options header declares no DeviceEntry")

    body = enum_match.group(1)
    enumerators = [m.group(1) for m in
                   re.finditer(r"^\s*(k[A-Za-z0-9_]+)\s*(?:=[^,]*)?,", body, re.MULTILINE)]

    if not enumerators or enumerators[-1] != "kCount":
        raise SystemExit("device_kernel_surface: DeviceEntry does not end in kCount")

    enumerators = enumerators[:-1]
    comments = {}

    for line in body.splitlines():
        found = re.match(r"\s*(k[A-Za-z0-9_]+)\s*(?:=[^,]*)?,", line)

        if found is not None:
            mark = line.find("///<", found.end())
            comments[found.group(1)] = line[mark + 4:].strip() if mark >= 0 else ""

    axes: dict[str, tuple[str, str, str, str]] = {}

    for cases, cells in switch_arms(function_body(text, "DeviceEntryAxesOf(DeviceEntry entry)")):
        if len(cells) != 4:
            raise SystemExit(f"device_kernel_surface: an arm of DeviceEntryAxesOf states "
                             f"{len(cells)} cells and not four: {cells}")

        named = [cell.rsplit("::", 1)[-1] for cell in cells]

        for case in cases:
            axes[case] = (named[0], named[1], named[2], cells[3])

    partitions: dict[str, str] = {}

    for cases, cells in switch_arms(function_body(text, "DevicePartitionOf(DeviceEntry entry)")):
        if len(cells) != 1:
            raise SystemExit(f"device_kernel_surface: an arm of DevicePartitionOf states "
                             f"{len(cells)} cells and not one")

        for case in cases:
            partitions[case] = cells[0].rsplit("::", 1)[-1]

    entries = []

    for order, name in enumerate(enumerators):
        if name not in axes or name not in partitions:
            raise SystemExit(f"device_kernel_surface: {name} is an enumerator no arm of "
                             f"DeviceEntryAxesOf or DevicePartitionOf names, so an enumerator can "
                             f"reach the space with no statement of its axes")

        comment = comments.get(name, "")
        surface = ""
        found = re.search(r"BoysCuda::(\w+)", comment)

        if found is not None:
            surface = found.group(1)
        else:
            found = re.search(r"BoysDevice(\w+)", comment)

            if found is not None:
                surface = "BoysDevice" + found.group(1)

        route, scheme, packing, division = axes[name]
        entries.append(Entry(name, order, route, scheme, packing, partitions[name], division,
                             surface, comment))

    return entries


class Row(NamedTuple):
    """One row of the library's own option table, as the cells this tool reads it by."""

    entry: str
    printed: str
    group: str
    precision: str
    shape: str
    question: str
    axis: str
    region: str
    lane: str
    built: str
    refused: str


def split_cells(body: str) -> list[str]:
    """A row's cells, split on the commas outside its string literals."""
    cells, current, in_string, escaped = [], [], False, False

    for ch in body:
        if in_string:
            current.append(ch)

            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False

            continue

        if ch == '"':
            in_string = True
            current.append(ch)
        elif ch == ",":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(ch)

    if current:
        cells.append("".join(current).strip())

    return cells


def read_rows(rows_source: Source) -> list[Row]:
    """Every row of `kDeviceOptions`, in the table's own order."""
    table = re.search(r"constexpr DeviceOptionInfo kDeviceOptions\[\]\s*=\s*\{(.*?)\n\};",
                      rows_source.text, re.DOTALL)

    if table is None:
        raise SystemExit("device_kernel_surface: the translation unit declares no kDeviceOptions "
                         "array, so the rows this tool walks are not in it")

    body = table.group(1)
    found = []
    index = 0

    while True:
        start = body.find("{", index)

        if start < 0:
            break

        position, in_string, escaped = start + 1, False, False

        while position < len(body):
            ch = body[position]

            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "}":
                break

            position += 1

        row = body[start + 1:position]
        index = position + 1
        cells = split_cells(row)

        if not cells or not cells[0].startswith("DeviceEntry::"):
            continue

        # entry, printed name, group, precision, shape, question, axis, regionBExp, lane, bound,
        # boundForm, built, refusedBecause.
        if len(cells) < 13:
            raise SystemExit(f"device_kernel_surface: the row {cells[0]} has {len(cells)} cells and "
                             f"a row of this table has thirteen. A row read short is a combination "
                             f"this tool would count as missing")

        found.append(Row(cells[0].rsplit("::", 1)[-1], cells[1].strip('"'),
                         cells[2].rsplit("::", 1)[-1], cells[3].rsplit("::", 1)[-1],
                         cells[4].rsplit("::", 1)[-1], cells[5].rsplit("::", 1)[-1],
                         cells[6].rsplit("::", 1)[-1], cells[7].rsplit("::", 1)[-1],
                         cells[8].rsplit("::", 1)[-1], cells[11], cells[12]))

    return found


class Surface(NamedTuple):
    """A function of one of the two surfaces, as its own declaration reads."""

    group: str
    name: str
    template: str
    params: str

    @property
    def takes_form(self) -> bool:
        """Whether a call of it can name a division form."""
        return "DivisionForm" in self.template or "DivisionForm" in self.params


def paren_span(text: str, start: int) -> int:
    """The index just past the parameter list that opens at `start`."""
    depth, position = 1, start

    while depth and position < len(text):
        if text[position] == "(":
            depth += 1
        elif text[position] == ")":
            depth -= 1

        position += 1

    return position - 1


def read_surfaces(launched: Source, inkernel: Source) -> dict[str, Surface]:
    """Every function of both surfaces, keyed by name.

    Read from the declarations, because what an entry's row states and what a call can name are two
    facts: an entry whose surface takes no division form is an entry the form axis cannot reach,
    whatever the table says about it.
    """
    surfaces: dict[str, Surface] = {}

    for match in re.finditer(r"(?:template\s*<([^;{}]*?)>\s*)?static\s+BoysStatus\s+(\w+)\s*\(",
                             launched.text):
        end = paren_span(launched.text, match.end())
        surfaces[match.group(2)] = Surface(LAUNCHED_GROUP, match.group(2),
                                           (match.group(1) or "").strip(),
                                           launched.text[match.end():end])

    for match in re.finditer(r"(?:template\s*<([^;{}]*?)>\s*)?__device__\s+BoysDeviceStatus\s+"
                             r"(\w+)\s*\(", inkernel.text):
        end = paren_span(inkernel.text, match.end())
        surfaces[match.group(2)] = Surface(DEVICE_GROUP, match.group(2),
                                           (match.group(1) or "").strip(),
                                           inkernel.text[match.end():end])

    if not surfaces:
        raise SystemExit("device_kernel_surface: neither surface declares a function, so no entry "
                         "could be reached by one")

    return surfaces


def read_kernel_tags(kernels: Source) -> dict[str, str]:
    """The tag-parameterised body each launched kernel hands its lane to, keyed by kernel name.

    A missing launched row is a bounded job where the kernel's body already exists; this is where
    that is read, on the file's own structure: a kernel is a wrapper over a tag-parameterised body.
    """
    tags: dict[str, str] = {}

    for match in re.finditer(r"__global__\s+void\s+(\w+)\s*\(", kernels.text):
        tail = kernels.text[match.end():match.end() + 4000]
        found = re.search(r"detail::(Device\w+)\s*<([^>]*)>", tail)

        if found is not None:
            tags[match.group(1)] = f"{found.group(1)}<{found.group(2)}>"

    return tags


def read_host_axes(probe: Source) -> list[str]:
    """The axis fields the host's own cell struct declares."""
    body = re.search(r"struct OptionProbeCell\s*\{(.*?)\n\};", probe.text, re.DOTALL)

    if body is None:
        raise SystemExit("device_kernel_surface: the probe header declares no OptionProbeCell, so "
                         "the host's axes cannot be read from the host's own statement")

    axes = []

    for line in body.group(1).splitlines():
        found = re.match(r"\s*(FitRoute|EvalScheme|FitGranularity|PackAxis|DivisionForm|RegionBExp)"
                         r"\s+(\w+)", line)

        if found is not None:
            axes.append(found.group(2))

    return axes


class Combination(NamedTuple):
    """One point of a class's axis space."""

    precision: str
    format: str
    shape: str
    group: str
    coords: tuple[tuple[str, str], ...]

    def member(self, axis: str) -> str:
        return dict(self.coords).get(axis, "")


def product_of(axes: tuple[tuple[str, list[str]], ...]) -> list[list[tuple[str, str]]]:
    """The product of the axes a shape admits, as a list of coordinate lists."""
    result: list[list[tuple[str, str]]] = [[]]

    for axis, members in axes:
        result = [partial + [(axis, member)] for partial in result for member in members]

    return result


def render(coords: tuple[tuple[str, str], ...]) -> str:
    """One combination, in one line."""
    short = {"kRoute": "route", "kScheme": "scheme", "kPartition": "part", "kPacking": "pack",
             "kRegionBExp": "exp", "kDivision": "form"}

    return " ".join(f"{short.get(axis, axis)}={member}" for axis, member in coords)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("head", "worktree"), default="head",
                        help="read the committed revision (default) or the working tree")
    parser.add_argument("--summary", action="store_true", help="print class totals only")
    parser.add_argument("--missing", action="store_true",
                        help="print the missing combinations, class by class")
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero when the enumeration is empty or a defect stands")
    args = parser.parse_args()

    revision, sources = read_sources(args.source)
    options = sources[OPTIONS]
    entries = read_entries(options)
    by_entry = {entry.name: entry for entry in entries}
    rows = read_rows(sources[ROWS])
    rows_by_entry = {row.entry: row for row in rows}
    surfaces = read_surfaces(sources[LAUNCHED], sources[INKERNEL])
    tags = read_kernel_tags(sources[KERNELS])

    print(f"device_kernel_surface: revision {revision} of {REPO}")
    print(f"  read with `git show <revision>:<path>`; a working tree is never read unless --source "
          f"worktree is named")
    named = {entry.surface for entry in entries}
    orphaned = sorted(name for name in surfaces if name not in named)
    print(f"  entries {len(entries)}   rows {len(rows)}   surfaces an entry names "
          f"{sum(1 for n, s in surfaces.items() if n in named)} "
          f"({sum(1 for n, s in surfaces.items() if n in named and s.group == LAUNCHED_GROUP)} "
          f"launched, "
          f"{sum(1 for n, s in surfaces.items() if n in named and s.group == DEVICE_GROUP)} "
          f"in-kernel)   kernels {len(tags)}")

    if orphaned:
        print(f"  declarations no entry names, so they carry no combination and are not surfaces "
              f"of this enumeration: {', '.join(orphaned)}")

    if not entries or not rows:
        print("device_kernel_surface: the enumeration read nothing, which is a failed run",
              file=sys.stderr)
        return 1

    precisions = enumerate_members(options.text, "DeviceOptionPrecision")
    shapes = enumerate_members(options.text, "DeviceOptionShape")
    questions = enumerate_members(options.text, "DeviceOptionQuestion")
    groups = enumerate_members(options.text, "DeviceOptionGroup")
    axes_declared = [member for member in enumerate_members(options.text, "DeviceOptionAxis")
                     if member != "kNone"]
    packing_members = [member for member in enumerate_members(options.text, "DevicePacking")
                       if member not in SENTINELS and member != "kNotApplicable"]
    env = sources[ENUMS].text
    routes = enumerate_members(env, "FitRoute")
    schemes = enumerate_members(env, "EvalScheme")
    partitions = enumerate_members(env, "FitGranularity")
    regions = enumerate_members(env, "RegionBExp")
    forms = enumerate_members(env, "DivisionForm")
    host_axes = read_host_axes(sources[PROBE])

    # The half lane's two formats, from the precision enumeration's own comment: "the fp16 entries
    # this build serves, and the bfloat16 ones it owes".
    formats = {"kFp16": ("fp16", "bfloat16")}

    print()
    print("THE DEVICE'S AXES, declared by DeviceOptionAxis and read from the types its members name")
    print(f"  declared axes {len(axes_declared)}: {', '.join(axes_declared)}")

    for axis, members, note in (
            ("kRegionBExp", regions, "RegionBExp - the shared type, boys/accuracy.hpp"),
            ("kPartition", partitions, "FitGranularity - the host's own type, not a device enum"),
            ("kPacking", packing_members, "DevicePacking; kNotApplicable is stated not to be a "
                                          "member of the axis"),
            ("kScheme", schemes, "EvalScheme"),
            ("kRoute", routes, "FitRoute"),
            ("kDivision", forms, "DivisionForm")):
        print(f"  {axis:<14} {len(members)} members  {members}\n"
              f"  {'':<14} ({note})")

    print(f"  The host's own cell struct declares {len(host_axes)} axes: {', '.join(host_axes)}.")
    print(f"  The device declares {len(axes_declared)}; the sets correspond member for member, so "
          f"the device's axes are the host's six and not the host's six minus one. What differs is "
          f"the vocabulary (DevicePacking's kLadder/kPerOrder against PackAxis's "
          f"kArguments/kOrders) and the scope of each axis's built rows.")

    # The exponential is the member a reader is likeliest to read as a property of the lane rather
    # than a coordinate: the library says otherwise, in the host cell's own words and in the device
    # half's fallback, so both are printed here with their lines.
    for statement in (
            ("the exponential is a coordinate of the option, on the same reading as the division "
             "form", "both members are served at every cell of the other axes, and the member is "
             "part of the combination an option runs", PROBE),
            ("a device class carries it as the device lane's own member",
             "on a class of the device half those four beside the device lane's own division form "
             "and region-B exponential, because an unnamed device call reads the device lane's two "
             "names and never the host's", PROBE)):
        sentence, quote, path = statement
        where = quote_line(sources[path].text, quote)

        if where == 0:
            print(f"device_kernel_surface: an axis statement cites a sentence {path} no longer "
                  f"carries: {quote!r}", file=sys.stderr)
            return 1

        print(f"  {sentence}:\n      {path}:{where}  \"{quote}\"")

    shape_axes = {
        "kSingle": (("kRegionBExp", regions), ("kDivision", forms)),
        "kAllOrders": (("kRoute", routes), ("kScheme", schemes), ("kPartition", partitions),
                       ("kPacking", packing_members), ("kRegionBExp", regions),
                       ("kDivision", forms)),
        "kAllN": (("kRegionBExp", regions), ("kDivision", forms)),
        "kEachOrder": (("kRegionBExp", regions), ("kDivision", forms)),
    }
    rules = (
        ("the single shape fixes no reading of the packing axis, so no member of it is enumerated "
         "for that shape",
         "one order asked for, one fit answering it, so no ladder is read and the packing axis has "
         "no member here", OPTIONS),
        ("the all-N and each-order shapes seed region B, so the exponential axis applies to them",
         "the all-N and each-order shapes seed the coarsest ladder at one top order", OPTIONS),
        ("the ladder shape's own four axes, with the region-B exponential beside the division form",
         "The three shapes per precision family: Single* (one order per argument), AllOrders* (all "
         "orders per argument, top order per element), AllN* (all orders at every argument, one "
         "common top order).", LAUNCHED),
        ("the division form is a coordinate of every entry of every shape and is never the reason "
         "a combination is missing",
         "Every entry of this space can run every form, so the form is a coordinate of the "
         "measurement and not a property of the entry", ENTRIES),
    )

    print()
    print("THE RULES THIS ENUMERATION APPLIES, each with the sentence it is read from")

    for rule, quote, path in rules:
        where = quote_line(sources[path].text, quote)

        if where == 0:
            print(f"device_kernel_surface: the rule `{rule}` cites a sentence {path} no longer "
                  f"carries: {quote!r}", file=sys.stderr)
            return 1

        print(f"  rule: {rule}")
        print(f"        {path}:{where}  \"{quote}\"")

    print()

    classes: dict[tuple[str, str, str, str], list[Combination]] = collections.OrderedDict()

    for precision in precisions:
        for fmt in formats.get(precision, ("",)):
            for shape in shapes:
                for group in groups:
                    space = [Combination(precision, fmt, shape, group,
                                         tuple((axis, member) for axis, member in coords))
                             for coords in product_of(shape_axes[shape])]
                    classes[(precision, fmt, shape, group)] = space

    def implemented_by(combination: Combination) -> list[Row]:
        """The class's rows that carry this combination's coordinates."""
        if combination.format == "bfloat16":
            return []

        found = []

        for row in rows:
            entry = by_entry.get(row.entry)

            if entry is None or row.precision != combination.precision:
                continue

            if row.shape != combination.shape or row.group != combination.group:
                continue

            values = {"kRegionBExp": row.region, "kPartition": entry.partition,
                      "kPacking": entry.packing, "kScheme": entry.scheme, "kRoute": entry.route}

            if all(values[axis] == member for axis, member in combination.coords if axis in values):
                found.append(row)

        return found

    # Whether each row's own surface takes a division form: a row whose entry names a function no
    # header declares is reported rather than counted, and a row whose surface takes none is one the
    # form axis cannot reach.
    form_capable = {}

    for row in rows:
        entry = by_entry.get(row.entry)
        surface = surfaces.get(entry.surface) if entry is not None else None
        form_capable[row.entry] = surface is not None and surface.takes_form

    impossible_quotes = (
        (lambda c: c.member("kPartition") == "kUniform" and c.member("kPacking") == "kLadder",
         "the grid has no ladder to read", OPTIONS),
        (lambda c: c.member("kPartition") == "kUniform" and c.member("kPacking") == "kLadder",
         "the grid's cells carry their own degree and block start, so the route's packing axis has "
         "one member", OPTIONS),
    )
    def scope_notes(combination: Combination) -> list[tuple[str, str]]:
        """The library's own statements about where this combination's members are built.

        A note is not a reason the combination is impossible: it is the scope the axis is built to
        today, which is a defect's context. Only a sentence the library writes *about the member
        this combination names* may be cited, so each note below names its subject.
        """
        notes: list[tuple[str, str]] = []

        if combination.format == "bfloat16":
            notes.append(("This build serves that lane's fp16 entries and not its bfloat16 ones.",
                          OPTIONS))
            notes.append(("Until that lands this member is served for fp16 alone, which is unbuilt "
                          "work and not a property of the lane", OPTIONS))

        if combination.member("kRegionBExp") == "kFast":
            if combination.group == DEVICE_GROUP:
                notes.append(("The float single entry carries one option the others do not: its "
                              "region-B exponential is a certified choice of arithmetic", INKERNEL))
            else:
                notes.append(("One entry carries a second axis on top of it: the f32 single "
                              "entry's region-B exponential", LAUNCHED))

        return notes

    totals = collections.Counter()
    defects: list[Combination] = []
    missing_by_class: collections.Counter = collections.Counter()
    implemented_count: dict[tuple[str, str, str, str], int] = {}

    for key, space in classes.items():
        precision, fmt, shape, group = key
        ok, missing = [], []
        have = 0

        for combination in space:
            rows_here = implemented_by(combination)

            if rows_here:
                have += 1
                totals["ok"] += 1
                names = ", ".join(sorted({row.entry for row in rows_here}))
                printed = ", ".join(sorted({row.printed for row in rows_here}))
                reachable = [row for row in rows_here if form_capable.get(row.entry, False)]
                ok.append(f"  ok        {render(combination.coords)}  -> {names}  ({printed})")

                if not reachable:
                    print(f"device_kernel_surface: {names} implements a combination whose surface "
                          f"takes no division form, so the form axis cannot reach it",
                          file=sys.stderr)
                    return 1

                continue

            verdict, why = "DEFECT", ""

            for predicate, quote, path in impossible_quotes:
                if predicate(combination):
                    where = quote_line(sources[path].text, quote)

                    if where == 0:
                        print(f"device_kernel_surface: an impossibility cites a sentence {path} no "
                              f"longer carries: {quote!r}", file=sys.stderr)
                        return 1

                    verdict, why = "IMPOSSIBLE", f'{path}:{where} "{quote}"'
                    break

            if verdict == "DEFECT":
                citations = []

                for quote, path in scope_notes(combination):
                    where = quote_line(sources[path].text, quote)

                    if where == 0:
                        print(f"device_kernel_surface: a note cites a sentence {path} no "
                              f"longer carries: {quote!r}", file=sys.stderr)
                        return 1

                    citations.append(f'{path}:{where} "{quote}"')

                if citations:
                    why = ("the scope this axis is built to today, in the library's words: "
                           + "; ".join(citations)
                           + " - a statement of scope, not a reason the combination is impossible")

            counterpart = ""
            witnesses: list[tuple[object, Row]] = []

            for other in entries:
                row = rows_by_entry.get(other.name)

                if row is None or row.shape != combination.shape:
                    continue

                values = {"kRoute": other.route, "kScheme": other.scheme,
                          "kPacking": other.packing, "kPartition": other.partition}

                if not all(values[axis] == member for axis, member in combination.coords
                           if axis in values):
                    continue

                if "kRegionBExp" in dict(combination.coords) \
                        and row.region != combination.member("kRegionBExp"):
                    continue

                witnesses.append((other, row))

            # A bfloat16 combination's nearest witness is the lane's own fp16 rows: the library's
            # own argument is that such a row is "a row over this lane's existing tables", the same
            # engine's value stored into bfloat16, so the format is the only difference left.
            if combination.format == "bfloat16":
                same_lane = [pair for pair in witnesses
                             if pair[1].precision == combination.precision]

                if same_lane:
                    witnesses = same_lane

            if witnesses:
                other, row = witnesses[0]
                conditions = []

                if row.group != combination.group:
                    conditions.append("the other route")

                if row.precision != combination.precision:
                    conditions.append(f"another lane ({row.precision})")
                elif combination.format == "bfloat16":
                    conditions.append("the fp16 format this lane serves")

                if conditions:
                    tag = tags.get("Boys" + other.surface + "Kernel", "")
                    counterpart = (f"can it be: yes - {other.name} carries this combination on "
                                   f"{' and '.join(conditions)}"
                                   + (f", over the body {tag}" if tag else ""))

            if not counterpart:
                counterpart = ("can it be: no counterpart on either route or any lane carries this "
                               "combination, and the library states no impossibility - owed work")

            missing.append(f"  MISSING   {render(combination.coords)}  {verdict}"
                           + (f"  [{why.strip()}]" if why.strip() else "")
                           + f"\n            {counterpart}")
            totals["missing"] += 1
            missing_by_class[key] += 1

            if verdict == "DEFECT":
                defects.append(combination)

        implemented_count[key] = have
        label = f"{precision} x {fmt + ' x ' if fmt else ''}{shape} x {group}"

        if args.missing:
            if missing:
                print(f"CLASS {label}   combinations {len(space)}   ok {have}   missing "
                      f"{len(missing)}")

                for line in missing:
                    print(line)

                print()
        else:
            print(f"CLASS {label}   combinations {len(space)}   ok {have}   missing {len(missing)}")

            if not args.summary:
                for line in ok:
                    print(line)

                for line in missing:
                    print(line)

                print()

    print("MISSING BY CLASS, the work item")
    print(f"  {'lane':<8} {'format':<9} {'shape':<11} {'route':<15} {'have':>5} {'of':>5} "
          f"{'missing':>8}")

    for key, space in classes.items():
        precision, fmt, shape, group = key
        have = implemented_count[key]
        print(f"  {precision:<8} {fmt or '-':<9} {shape:<11} {group:<15} {have:>5} {len(space):>5} "
              f"{len(space) - have:>8}")

    print()
    print(f"TOTALS  classes {len(classes)}   combinations {totals['ok'] + totals['missing']}   "
          f"implemented {totals['ok']}   missing {totals['missing']}   of those defects "
          f"{len(defects)}")

    if not classes or totals["ok"] + totals["missing"] == 0:
        print("device_kernel_surface: the enumeration walked no combination, which is a failed run",
              file=sys.stderr)
        return 1

    if args.check and defects:
        print(f"device_kernel_surface: {len(defects)} combination(s) the surface can express and no "
              f"entry implements, with no impossibility stated", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
