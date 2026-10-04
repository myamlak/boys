#!/usr/bin/env python3
r"""Hold every class, and every combination, to a default a consumer can actually be given.

WHY. A call site that names no policy resolves `DefaultPolicy<Precision, Shape>` - a row of the
build-defaults seam (`include/boys/boys_build_defaults.hpp`). Four things have to hold of a
combination the library serves before a consumer can be handed it as a default, and the first
three are checked elsewhere:

  1. the library carries the combination - `tools/check_class_surface.py`,
     `tools/check_device_class_surface.py`, `tools/check_host_space_closes.py`;
  2. the combination is measured - `tools/check_probe_measures_every_entry.py`,
     `tools/check_default_rows_have_entries.py`;
  3. the combination is offerable at a call site - `tools/check_enum_arms_covered.py`,
     `tools/combination_matrix.py`.

The fourth is this file's: **can the combination BE the consumer's default at all** - is there a
row of the seam's key for the class, and does the row format have a cell for each axis of the
combination? A combination the library serves, measures and probes, but which cannot be written
as a default, is a defect by the owner's rule; a class with no row at all is one too.

WHAT IS READ, AND FROM WHERE. Nothing below is a list written into this file. The classes the
seam's key reaches are read from the library's own two row lists (`BOYS_DEFAULT_POLICY_BUILD_ROWS`
and `..._DEVICE_ROWS`, `boys/boys.hpp`), which state themselves as "the entries' own classes, read
off the entries"; the entries' policy defaults corroborate the host half, and the device half is
corroborated against `DeviceOptionPrecision x DeviceOptionQuestion` (`boys_cuda_options.hpp`). The
seam's rows come from `BOYS_BUILD_DEFAULT_ROWS`. The axes of a policy come from the parameter list
of `StatedEvalPolicy` and the enumerators of each axis's own header, found by name over
`include/boys/*.hpp`. The measurement's own key - the classes a probe run ranks - comes from the
probe's source: `kCellPrecisions`, `kSeamShapes`, `kSeamDeviceShapes` and `LaneOf` in
`src/boys_probe.cpp`.

THE HOLE THIS EXISTS TO NAME. `Precision::kFp16` is documented as "half precision: the fp16 and
bfloat16 entries" (`boys/boys.hpp`) - one lane, two formats, one key. The probe keys them as two
classes (`OptionPrecision::kFp16` and `kBf16`) and folds both onto the one lane (`LaneOf`), so
where the two classes' measured winners differ, one of the two cannot be captured as a default:
the lane has one row and the format that does not own it has nowhere to be written. This check
reads the fold off the probe's own source, names the class that cannot be captured, and - when a
run's report is handed in with `--report` - holds the fold to the report's own verdict on the two
half formats rather than to the prose in the seam.

NO PROBE IS RUN. Every reading is textual, from the tree in front of this file.

WHAT IS NOT CHECKED HERE. Whether a row is *reachable* - an entry or an asking unit naming the
class - is `tools/check_default_rows_have_entries.py`'s, and whether an entry's body can carry the
combination is the accuracy gate's. This file reads the key, the format and the report, and
nothing else.

Usage:

    python tools/check_default_capturability.py [--check] [--report RUN.txt]
    python tools/check_default_capturability.py [--root PATH]     # an export of HEAD, say
    python tools/check_default_capturability.py --space kHost,kFp64,kAllOrders

`--root` reads another tree - an export of HEAD, a build tree - while the tool itself stays where
it is. A claim about what the library contains is read from a tree no writer holds: the working
tree of a busy checkout is not one, and an export of the commit is.

Exit status is 0 when every class of the seam's key carries a row, every axis of a policy has a
cell in the row format, and every class the measurement keys has a cell of its own. It is 1
otherwise - and 1 when the reading found nothing to check, because a run that checks nothing must
never read as a pass.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCLUDE = os.path.join(REPO, "include", "boys")

SEAM_FILE = os.path.join(INCLUDE, "boys_build_defaults.hpp")
BOYS_FILE = os.path.join(INCLUDE, "boys.hpp")
PROBE_SRC = os.path.join(REPO, "src", "boys_probe.cpp")
CUDA_FILE = os.path.join(INCLUDE, "boys_cuda.hpp")

# The macro the seam's row list is written under, and the macro `boys.hpp` expands a row through.
SEAM_ROWS_MACRO = "BOYS_BUILD_DEFAULT_ROWS"
SEAM_ROW_MACRO = "BOYS_DEFAULT_POLICY_ROW"
KEY_ROW_MACROS = (
    "BOYS_DEFAULT_POLICY_BUILD_ROWS",
    "BOYS_DEFAULT_POLICY_BUILD_DEVICE_ROWS",
)

# The seven names a seam states beside its rows: the host's five and the device lane's two. Read
# from the file by name, so a seam that stops defining one is an error here rather than a hole.
SEAM_NAMES = {
    "route": "BOYS_BUILD_DEFAULT_FIT_ROUTE",
    "scheme": "BOYS_BUILD_DEFAULT_EVAL_SCHEME",
    "pack": "BOYS_BUILD_DEFAULT_PACK_AXIS",
    "granularity": "BOYS_BUILD_DEFAULT_FIT_GRANULARITY",
    "division": "BOYS_BUILD_DEFAULT_DIVISION_FORM",
    "deviceDivision": "BOYS_BUILD_DEFAULT_DEVICE_DIVISION_FORM",
    "deviceExp": "BOYS_BUILD_DEFAULT_DEVICE_REGION_B_EXP",
}


# --------------------------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------------------------


def read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def blank(text: str) -> str:
    """The text with its comments blanked out, line structure and length kept.

    A name is read out of the code and not out of what the code says about itself: every header
    here spells the row format out in prose - `X(kHost, kFp64, ...)`, `DefaultPolicy<Precision::
    kX, Shape::kY>` - and a reader that does not skip comments books rows and classes that no
    declaration carries. Blanking rather than dropping keeps the line numbers a report prints the
    file's own.
    """
    out: list[str] = []
    index = 0
    while index < len(text):
        pair = text[index:index + 2]
        if pair == "//":
            end = text.find("\n", index)
            end = len(text) if end < 0 else end
            out.append(" " * (end - index))
            index = end
        elif pair == "/*":
            end = text.find("*/", index + 2)
            end = len(text) if end < 0 else end + 2
            out.append("".join("\n" if c == "\n" else " " for c in text[index:end]))
            index = end
        else:
            out.append(text[index])
            index += 1
    return "".join(out)


def strip_comments(text: str) -> str:
    """One cell's text with any comment removed - the cells are short, so no line structure."""
    return blank(text).strip()


def line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def headers() -> list[str]:
    return sorted(glob.glob(os.path.join(INCLUDE, "*.hpp")))


def enumerators(name: str) -> tuple[list[str], str, str]:
    """The members of `enum class name`, its header, and the header's own text.

    A member whose own documentation says it is "one past the last" is a count sentinel and not a
    member: the device surface's two enumerations carry one so a loop has a bound, and a sentinel
    counted as a member would answer every coverage question with one class too many.

    The name is matched as a declaration and not as a word: `Device` is a prefix of
    `DeviceOptionPrecision`, and a reader that took the first hit would read the device surface's
    clauses for the lane key. Where a name is declared in more than one header, or in none, this
    returns nothing and the caller reports the read rather than a member list.
    """
    pattern = re.compile(r"enum class\s+" + re.escape(name) + r"\s*[:{<]")
    hits: list[tuple[str, str, "re.Match[str]"]] = []
    for path in headers():
        text = blank(read(path))
        found = pattern.search(text)
        if found is not None:
            hits.append((path, text, found))
    if len(hits) != 1:
        return [], "", ""
    path, text, head = hits[0]
    brace = text.find("{", head.start())
    if brace < 0:
        return [], "", ""
    end = text.find("};", brace)
    body = text[brace + 1:end if end >= 0 else len(text)]

    members: list[str] = []
    for part in split_top_level(body):
        match = re.match(r"\s*(k[A-Za-z0-9_]+)", part)
        if match is None:
            continue
        doc = re.search(r"///<(.*)", part)
        said = doc.group(1) if doc else ""
        if "one past the last" in said or match.group(1) == "kCount":
            continue
        members.append(match.group(1))
    return members, path, text


def bare(cell: str) -> str:
    """A cell's member name without its enumeration: `FitRoute::kChebyshev` -> `kChebyshev`."""
    return cell.split("::")[-1].strip()


def find_macro(text: str, name: str) -> tuple[str, int] | None:
    """The expansion of `#define name(...)` and where it starts, comments and breaks left in.

    The comments are kept because the seam's rows carry their marker in one: whether a row is a
    measurement or one of the file's own names is written above the row and nowhere else. The
    backslash-newlines are kept too, so the text this returns is the file's own and a byte in it
    and a position in the file differ by the offset returned beside it.

    The scan is the compiler's, and not "every line but the last ends in a backslash": a comment
    inside a definition may span lines without one, and the seam uses that - the block noting that
    the half lane is one lane for both formats is written across four lines with a backslash only
    on the last, and a reader that stopped at the first of them would lose the nine device rows
    that follow it and report a seam of fifteen classes for a seam of twenty-four.
    """
    match = re.search(r"^#define\s+" + re.escape(name) + r"\s*\(", text, re.M)
    if match is None:
        return None
    start = match.start()
    index = start
    end = len(text)
    in_block_comment = False
    in_line_comment = False
    while index < len(text):
        pair = text[index:index + 2]
        if in_block_comment:
            if pair == "*/":
                in_block_comment = False
                index += 2
                continue
        elif in_line_comment:
            if text[index] == "\n":
                in_line_comment = False
        elif pair == "/*":
            in_block_comment = True
            index += 2
            continue
        elif pair == "//":
            in_line_comment = True
            index += 2
            continue
        elif text[index] == "\\":
            step = 2 if text[index + 1:index + 2] == "\r" else 1
            if text[index + step:index + step + 1] == "\n":
                index += step + 1
                continue
        elif text[index] == "\n":
            end = index + 1
            break
        index += 1
    return text[start:end], start


def splice(block: str) -> tuple[str, list[int]]:
    """A macro's expansion with its line continuations joined, and a map back to the raw index.

    A row is written over several lines, each ending in a backslash, so a cell's text is not
    contiguous in the file until the continuations are joined - which is what the preprocessor
    does with it, and what a reader that skipped the joining would take for a cell named
    `\\\n      PackAxis::kArguments`.

    The line break after the backslash is one or two characters: the same file is checked out with
    LF on one host and CRLF on another, and a reader that joined only one of the two would read a
    cell's text as carrying a line break on the other.
    """
    joined: list[str] = []
    index_map: list[int] = []
    index = 0
    while index < len(block):
        if block[index] == "\\":
            step = 2 if block[index + 1:index + 2] == "\r" else 1
            if block[index + step:index + step + 1] == "\n":
                joined.append(" ")
                index_map.append(index)
                index += step + 1
                continue
        joined.append(block[index])
        index_map.append(index)
        index += 1
    return "".join(joined), index_map


def matching_paren(text: str, at: int) -> int:
    depth = 0
    index = at
    while index < len(text):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def split_top_level(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return parts


def comment_spans(text: str) -> list[tuple[int, int]]:
    """The half-open ranges of the text that are a comment, so a call written in prose is not a row.

    The seam's rows are documented by example above the list - a row format is easier to state with
    one in it - and a reader that took every `X(` for a row would book a class the library does not
    reach, or a row the seam does not carry.
    """
    spans = [(match.start(), match.end()) for match in re.finditer(r"/\*.*?\*/", text, re.S)]
    spans += [(match.start(), match.end()) for match in re.finditer(r"//[^\n]*", text)]
    return spans


def in_span(spans: list[tuple[int, int]], at: int) -> bool:
    return any(start <= at < end for start, end in spans)


def invocations(block: str, macro: str = "X",
                spans: list[tuple[int, int]] | None = None) -> list[tuple[list[str], str, int]]:
    """Every `macro(...)` call in a block, as (cells, the text before it, where it starts).

    The text before each call is what carries the seam's marker comment, and it is taken from the
    end of the previous call so that a comment belonging to a row cannot be read as belonging to
    the next one. The position is an index into the block as given: a block whose continuations
    have been joined needs the map `splice` returns to put a row back where the file wrote it.
    """
    found: list[tuple[list[str], str, int]] = []
    search = macro + "("
    index = 0
    previous_end = 0
    while True:
        at = block.find(search, index)
        if at < 0:
            break
        if at > 0 and (block[at - 1].isalnum() or block[at - 1] == "_"):
            index = at + len(search)
            continue
        if spans and in_span(spans, at):
            index = at + len(search)
            continue
        end = matching_paren(block, at + len(macro))
        if end < 0:
            break
        inside = block[at + len(search):end]
        cells = [strip_comments(cell) for cell in split_top_level(inside)]
        found.append((cells, block[previous_end:at], at))
        previous_end = end + 1
        index = end + 1
    return found


def comments_in(text: str) -> list[str]:
    """Every `/* ... */` block in a run of text, in order."""
    return [match.group(1) for match in re.finditer(r"/\*(.*?)\*/", text, re.S)]


def marker_of(gap: str) -> tuple[str, list[str]]:
    """A row's marker: the last comment block before it, and the blocks before that.

    The last block is the row's own; an earlier one is a note about the file or the lane the row
    belongs to, and those are printed beside the row rather than read as its marker.
    """
    blocks = comments_in(gap)
    if not blocks:
        return "", []
    return " ".join(blocks[-1].split()), [" ".join(block.split()) for block in blocks[:-1]]


# --------------------------------------------------------------------------------------------
# the seam
# --------------------------------------------------------------------------------------------


class Row:
    """One row of the seam: its class, its cells, and what the file says it is."""

    def __init__(self, cells: list[str], marker: str, notes: list[str], line: int, source: str):
        self.cells = cells
        self.marker = marker
        self.notes = notes
        self.line = line
        self.source = source

    @property
    def klass(self) -> tuple[str, str, str]:
        return (self.cells[0], self.cells[1], self.cells[2])


def seam_rows() -> tuple[list[Row], dict[str, str]]:
    text = read(SEAM_FILE)
    found = find_macro(text, SEAM_ROWS_MACRO)
    rows: list[Row] = []
    if found is not None:
        block, offset = found
        joined, index_map = splice(block)
        spans = comment_spans(joined)
        for cells, gap, at in invocations(joined, spans=spans):
            if len(cells) < 3:
                continue
            marker, notes = marker_of(gap)
            rows.append(Row(cells, marker, notes,
                            line_of(text, offset + index_map[at]), SEAM_FILE))

    names: dict[str, str] = {}
    for key, macro in SEAM_NAMES.items():
        match = re.search(r"^#define\s+" + re.escape(macro) + r"\s+(\S+)\s*$", blank(text), re.M)
        names[key] = match.group(1) if match else ""
    return rows, names


def constant_value(name: str) -> str:
    """The value a `inline constexpr <type> <name> = ...;` carries, from whichever header has it.

    The host's region-B exponential has no macro in the seam - the seam names the device lane's
    and the host's is `kDefaultHostRegionBExp` in `boys/accuracy.hpp` - so the comparison of a
    row's last cell against the choice an unnamed host call runs reads that constant, not a value
    written here.
    """
    pattern = re.compile(r"inline constexpr\s+[A-Za-z_][A-Za-z0-9_:<>]*\s+" + re.escape(name)
                         + r"\s*=\s*([^;]+);")
    for path in headers():
        match = pattern.search(blank(read(path)))
        if match is not None:
            return match.group(1).strip()
    return ""


def key_rows() -> list[tuple[str, str, str]]:
    """The seam's key as the library states it: the classes its entries reach, list by list.

    `boys.hpp` writes the host's classes out where the seam carries no row list, and says of them
    that they are "the entries' own classes, read off the entries" - so this is the library's own
    statement of the key rather than a second list beside the entries, and the device half's nine
    are written out the same way.
    """
    text = blank(read(BOYS_FILE))
    classes: list[tuple[str, str, str]] = []

    for macro in KEY_ROW_MACROS:
        # The list's name says which half it is: the host's list is the plain one, and the lane
        # whose entries take no policy carries "DEVICE" in its list's name.
        device = "kDevice" if "DEVICE" in macro else "kHost"
        found = find_macro(text, macro)
        if found is None:
            continue
        block, _offset = found
        joined, _index_map = splice(block)
        spans = comment_spans(joined)
        for cells, _gap, _at in invocations(joined, spans=spans):
            if len(cells) >= 2:
                classes.append((device, cells[0], cells[1]))
    return classes


def declared_name(text: str, at: int) -> str:
    """The name of the declaration a default belongs to, read forward from it.

    An entry writes its default in the template parameter list -
    `template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp64, Shape::kSingle>> double
    BoysSingle(int, double)` - so the name is the last identifier before the declaration's own
    `(`. A default that is not followed by a declaration within a paragraph is a use, not an
    entry, and reads as `?` rather than borrowing a name from further down the file.
    """
    head = text[at:at + 400]
    paren = head.find("(")
    if paren < 0:
        return "?"
    names = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", head[:paren])
    return names[-1] if names else "?"


def entry_classes() -> dict[tuple[str, str, str], list[str]]:
    """The class each entry's policy parameter defaults to, with the entries' names.

    The default is what a call site that names no policy resolves, so this is the key read from
    the other end: the entries themselves. A bf16 entry is one of these like any other - its
    default names the fp16 lane's class, which is the fold this file exists to name.
    """
    found: dict[tuple[str, str, str], list[str]] = {}

    for path in (BOYS_FILE, os.path.join(INCLUDE, "boys_span.hpp")):
        text = blank(read(path))
        for match in re.finditer(r"DefaultPolicy<\s*(?:[A-Za-z_][A-Za-z0-9_]*::)*Precision::(k[A-Za-z0-9]+)"
                                 r"\s*,\s*(?:[A-Za-z_][A-Za-z0-9_]*::)*Shape::(k[A-Za-z0-9]+)"
                                 r"(?:\s*,\s*(?:[A-Za-z_][A-Za-z0-9_]*::)*Device::(k[A-Za-z0-9]+))?\s*>",
                                 text):
            device = match.group(3) or "kHost"
            name = declared_name(text, match.end())
            key = (device, match.group(1), match.group(2))
            if name not in found.setdefault(key, []):
                found[key].append(name)
    return found


def policy_axes() -> list[tuple[str, str]]:
    """The axes of a policy, in the order `StatedEvalPolicy` takes them: (type, parameter)."""
    text = blank(read(os.path.join(INCLUDE, "backend.hpp")))
    head = re.search(r"template\s*<([^>]*)>\s*struct\s+StatedEvalPolicy", text)
    if head is None:
        return []
    axes: list[tuple[str, str]] = []
    for part in split_top_level(head.group(1)):
        part = part.strip()
        if not part:
            continue
        cells = part.split()
        if len(cells) == 2:
            axes.append((cells[0], cells[1]))
    return axes


def probe_key() -> dict[str, object]:
    """The classes a run of the option probe ranks, read from the probe's own source.

    The probe's key is not the seam's: it keys the bf16 entries as a class of their own and folds
    them onto the fp16 lane (`LaneOf`), and it walks its own shape list beside the seam's. Reading
    both from the source is what lets this check say which of the two keys is the coarser.
    """
    text = blank(read(PROBE_SRC))
    result: dict[str, object] = {"cells": [], "seamShapes": [], "deviceShapes": [], "laneOf": {}}

    for key, name in (("cells", "kCellPrecisions"), ("seamShapes", "kSeamShapes"),
                      ("deviceShapes", "kSeamDeviceShapes")):
        match = re.search(r"constexpr\s+[A-Za-z_]+\s+" + re.escape(name) + r"\s*\[\]\s*=\s*\{([^}]*)\}",
                          text)
        if match is not None:
            result[key] = [cell.strip().split("::")[-1]
                           for cell in split_top_level(match.group(1)) if cell.strip()]

    body = re.search(r"LaneOf\s*\([^)]*\)[^{]*\{(.*?)\n\}", text, re.S)
    if body is not None:
        result["laneOf"] = lane_fold(body.group(1), text[body.end():body.end() + 200])
    return result


def lane_fold(cases: str, tail: str) -> dict[str, str]:
    """The class -> lane fold a `switch` states, read the way the compiler reads it.

    A run of `case` labels shares the statement under it, and a run that only `break`s takes the
    function's own trailing `return`. `LaneOf` is written in both shapes - `case kFp16:` and
    `case kBf16:` share one return, and `case kFp64:` breaks to the return after the switch - so a
    reader that pairs one label with one `return` loses the fold's own default.
    """
    fold: dict[str, str] = {}
    pending: list[str] = []
    broken: list[str] = []
    after_break = cases
    for match in re.finditer(r"case\s+OptionPrecision::(k[A-Za-z0-9_]+)\s*:"
                             r"|return\s+Precision::(k[A-Za-z0-9_]+)\s*;"
                             r"|\bbreak\s*;", cases):
        if match.group(1):
            pending.append(match.group(1))
        elif match.group(2):
            for label in pending:
                fold[label] = match.group(2)
            pending = []
        else:
            broken.extend(pending)
            pending = []
            after_break = cases[match.end():]
    trailing = re.search(r"return\s+Precision::(k[A-Za-z0-9_]+)\s*;", after_break)
    if trailing is None:
        trailing = re.search(r"return\s+Precision::(k[A-Za-z0-9_]+)\s*;", tail)
    if trailing is not None:
        for label in broken:
            fold[label] = trailing.group(1)
    return fold


def kebab(name: str) -> str:
    """`kAllOrders` -> `all-orders`: the spelling a run's report prints a class in."""
    body = name[1:] if name.startswith("k") else name
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", body).lower()


def report_leaders(text: str, formats: list[str], klass: str) -> dict[str, tuple[str, str]]:
    """The winning entry a run's report names for each format, in the class the report flags.

    A report states a class's winner twice over: the refinement section votes over the entries a
    class left tied, and the summary line names the fastest where nothing was tied. Both are read,
    and each is labelled for what it is - the verdict the probe prints compares the voted leaders,
    so a fastest read as a leader would be a claim the report does not make.
    """
    shape = kebab(klass)
    leaders: dict[str, tuple[str, str]] = {}
    for fmt in formats:
        stem = r"^\s+" + re.escape(kebab(fmt)) + r"\s+" + re.escape(shape)
        head = re.search(stem + r"\s*:\s*\d+\s+run\(s\)", text, re.M)
        vote_entry = ""
        fastest = None
        if head is not None:
            block = text[head.end():]
            stop = re.search(r"^\s{2,4}\S+\s+\S+\s*:\s*\d+\s+run\(s\)", block, re.M)
            if stop is not None:
                block = block[:stop.start()]
            at = re.search(r"^\s+vote:", block, re.M)
            if at is not None:
                line = block[at.start():].splitlines()[0]
                rest = line.split("vote:", 1)[1].strip()
                entry, _space, reason = rest.partition(" ")
                vote_entry = entry
                reason = reason.strip().lstrip("-— ").split(",")[0].strip()
                leaders[kebab(fmt)] = (vote_entry, f"voted: {reason}")
            fastest = re.search(r"\bfastest\s+(\S+)\s+at\s+([\d.]+)\s*ns", block)
        if vote_entry:
            continue
        summary = re.search(stem + r"\s+\d+\s+\S+\s*\|\s*fastest\s+(\S+)\s+at\s+([\d.]+)\s*ns",
                            text, re.M)
        if summary is not None:
            leaders[kebab(fmt)] = (summary.group(1),
                                   f"fastest of the class, {summary.group(2)} ns/argument, and not "
                                   "the vote")
        elif fastest is not None:
            leaders[kebab(fmt)] = (fastest.group(1),
                                   f"fastest of the class, {fastest.group(2)} ns/argument, and not "
                                   "the vote")
    return leaders


def combination_of(entry: str, formats: list[str]) -> str:
    """An entry's name with its format suffix taken off.

    Two formats' entries are one combination when their names agree once the format is off - the
    names differ by construction, since each carries its own format in its last field.
    """
    for fmt in formats:
        suffix = "-" + kebab(fmt)
        if entry.endswith(suffix):
            return entry[: -len(suffix)]
    return entry


def device_key() -> list[tuple[str, str]]:
    precisions, _, _ = enumerators("DeviceOptionPrecision")
    questions, _, _ = enumerators("DeviceOptionQuestion")
    return [(precision, question) for precision in precisions for question in questions]


def device_call_default() -> tuple[str, str, int, int, int]:
    """What a device call that names no policy gets, read off the lane's own declarations.

    The device entries take no policy parameter: the form is a parameter defaulted to the seam's
    device name, the exponential is a template default of the lane's own constant, and every other
    axis is the entry's body. The three counts are the number of device declarations that spell
    each, so that a lane that stopped defaulting one of them is a count and not a silence.
    """
    text = blank(read(CUDA_FILE))
    forms = re.findall(r"=\s*kDefaultDeviceDivisionForm\b", text)
    exps = re.findall(r"=\s*kDefaultRegionBExp\b", text)
    policies = re.findall(r"DefaultPolicy\s*<", text)
    return "kDefaultDeviceDivisionForm", "kDefaultRegionBExp", len(forms), len(exps), len(policies)


# --------------------------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------------------------


def marker_kind(marker: str) -> str:
    """What a row's own marker says it is. A marker nothing recognises is reported as unknown."""
    if not marker:
        return "none"
    lowered = marker.lower()
    if "a choice, not a measurement" in lowered or "not the winner of a comparison" in lowered:
        return "choice"
    if "measured" in lowered:
        return "measured"
    if ("no probe run" in lowered or "the five above" in lowered or "the shipped five" in lowered
            or "the five macros above" in lowered):
        return "fallback"
    return "unknown"


def row_states_seam_names(row: Row, names: dict[str, str], device: bool,
                          host_exp: str) -> tuple[bool, list[str]]:
    """Whether a row's cells are the choices the build states, and where they differ.

    The budget cell is excluded on purpose: a lane's budget is the lane's and not the file's - the
    half lanes compile the half budget and every other lane the float one - so comparing it to a
    name here would report a difference the file never made. What is compared is the six axes a
    build states a value for: the five host macros, or the device lane's two names, and the
    region-B exponential each half's unnamed call runs. A row whose cells differ from all of them
    is a row carrying a measurement - which is what a row of a class a probe ranked is, and what
    the row's own marker says beside it.
    """
    if len(row.cells) < 10:
        return False, ["the row has fewer than ten cells"]
    want = {
        3: names.get("route", ""),
        4: names.get("scheme", ""),
        6: names.get("pack", ""),
        7: names.get("granularity", ""),
        8: names.get("deviceDivision" if device else "division", ""),
        9: names.get("deviceExp", "") if device else host_exp,
    }
    differing = [f"cell {cell} is {row.cells[cell]} and the choice an unnamed call runs is {name}"
                 for cell, name in sorted(want.items()) if name and row.cells[cell] != name]
    return not differing, differing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="check the tree (the default; accepted so CI can spell it)")
    parser.add_argument("--root", metavar="PATH", default="",
                        help="read another tree (an export of HEAD) instead of this file's own; "
                             "the tool itself stays where it is")
    parser.add_argument("--report", metavar="PATH", default="",
                        help="a run's report, to hold the half-format fold to what it says")
    parser.add_argument("--space", metavar="DEVICE,PRECISION,SHAPE", default="",
                        help="print one class's whole combination space and its verdict")
    args = parser.parse_args()

    global REPO, INCLUDE, SEAM_FILE, BOYS_FILE, PROBE_SRC, CUDA_FILE
    if args.root:
        REPO = os.path.abspath(args.root)
        INCLUDE = os.path.join(REPO, "include", "boys")
        SEAM_FILE = os.path.join(INCLUDE, "boys_build_defaults.hpp")
        BOYS_FILE = os.path.join(INCLUDE, "boys.hpp")
        PROBE_SRC = os.path.join(REPO, "src", "boys_probe.cpp")
        CUDA_FILE = os.path.join(INCLUDE, "boys_cuda.hpp")

    failures: list[str] = []
    notes: list[str] = []
    read_files = [SEAM_FILE, BOYS_FILE, PROBE_SRC, CUDA_FILE]

    # ---- the key, from the library ---------------------------------------------------------
    key = key_rows()
    rows, names = seam_rows()
    named_by_entries = entry_classes()

    if not key or not rows or not names.get("route"):
        print("check_default_capturability: the reading found nothing to check - "
              f"{len(key)} key class(es), {len(rows)} row(s), the seam's names "
              f"{'read' if names.get('route') else 'NOT read'}. A reader that fails is not a tree "
              "that passes.", file=sys.stderr)
        return 1

    axes = policy_axes()
    if not axes:
        print("check_default_capturability: no axes read from StatedEvalPolicy in backend.hpp - "
              "the row format cannot be held to the policy it writes, so nothing here is checked",
              file=sys.stderr)
        return 1

    axis_members: dict[str, list[str]] = {}
    for axis_type, _param in axes:
        members, path, _text = enumerators(axis_type)
        if not members:
            print(f"check_default_capturability: the axis {axis_type}, named by StatedEvalPolicy, "
                  "has no enumeration this reader can find under include/boys - the read is wrong, "
                  "not the library", file=sys.stderr)
            return 1
        axis_members[axis_type] = members
        read_files.append(path)

    device_members, device_path, _ = enumerators("Device")
    lane_members, lane_path, _ = enumerators("Precision")
    shape_members, shape_path, _ = enumerators("Shape")
    read_files += [device_path, lane_path, shape_path, os.path.join(INCLUDE, "boys_span.hpp"),
                   os.path.join(INCLUDE, "accuracy.hpp")]

    host_exp = constant_value("kDefaultHostRegionBExp")
    if not host_exp:
        failures.append("the host's region-B exponential could not be read from the header that "
                        "carries it (kDefaultHostRegionBExp): a host row's last cell cannot be "
                        "held to the choice an unnamed host call runs, and a comparison that "
                        "silently skips a cell is not a comparison")

    probe = probe_key()
    read_files.append(PROBE_SRC)
    device_classes = device_key()
    read_files.append(os.path.join(INCLUDE, "boys_cuda_options.hpp"))

    host_key = [k for k in key if k[0] == "kHost"]
    device_key_rows = [k for k in key if k[0] == "kDevice"]

    print("check_default_capturability: " + f"{len(set(read_files))} file(s) read: "
          + ", ".join(sorted(os.path.relpath(p, REPO) for p in set(read_files) if p)))
    print()
    print(f"THE SEAM'S KEY, as boys.hpp states it: {len(host_key)} host class(es), "
          f"{len(device_key_rows)} device class(es)")
    print(f"  host   {len(host_key):2d}  {', '.join(sorted({k[1] for k in host_key}))}"
          f"  x  {', '.join(sorted({k[2] for k in host_key}))}")
    print(f"  device {len(device_key_rows):2d}  "
          f"{', '.join(sorted({k[1] for k in device_key_rows}))}"
          f"  x  {', '.join(sorted({k[2] for k in device_key_rows}))}")

    # Corroboration: the entries name their own classes, and the device key is a cross.
    entry_keys = {k for k in named_by_entries if k[0] == "kHost"}
    if entry_keys != set(host_key):
        only_key = sorted(set(host_key) - entry_keys)
        only_entry = sorted(entry_keys - set(host_key))
        failures.append("the library's host class list and the entries' own policy defaults "
                        f"disagree: {len(only_key)} class(es) no entry defaults to "
                        f"({only_key[:3]}), {len(only_entry)} class(es) no row list names "
                        f"({only_entry[:3]})")
    device_named = {(precision, question) for precision, question in device_classes}
    seam_device_pairs = {(k[1].replace("Device", ""), k[2]) for k in device_key_rows}
    if seam_device_pairs != device_named:
        failures.append("the device class cross (DeviceOptionPrecision x DeviceOptionQuestion) and "
                        f"the device rows' classes disagree: {len(device_named - seam_device_pairs)} "
                        f"cross cell(s) with no row, {len(seam_device_pairs - device_named)} row(s) "
                        "of no cross cell")
    print(f"  corroborated by the entries' own policy defaults ({len(entry_keys)} host class(es)) "
          f"and by the device cross ({len(device_named)} cell(s))")
    print()

    # ---- the row format --------------------------------------------------------------------
    row_format = find_macro(blank(read(BOYS_FILE)), SEAM_ROW_MACRO)
    cells_named: list[str] = []
    if row_format is not None:
        row_macro, _row_offset = row_format
        head = re.search(r"#define\s+" + re.escape(SEAM_ROW_MACRO) + r"\s*\(([^)]*)\)",
                         row_macro)
        if head is not None:
            cells_named = [cell.strip() for cell in split_top_level(head.group(1))]

    if not cells_named:
        failures.append(f"the row format's cells could not be read from {SEAM_ROW_MACRO} in "
                        "boys.hpp, so nothing below is held to the format it writes")
    else:
        print(f"THE ROW FORMAT, as {SEAM_ROW_MACRO} takes it: {len(cells_named)} cell(s) - "
              + ", ".join(cells_named))
        print(f"  the key cells: {cells_named[0]}, {cells_named[1]}, {cells_named[2]}")
        print(f"  the axes of StatedEvalPolicy: "
              + ", ".join(f"{kind} {param}" for kind, param in axes))
        if len(cells_named) - 3 != len(axes):
            failures.append(f"the row format takes {len(cells_named) - 3} axis cell(s) and a policy "
                            f"has {len(axes)} axes: the two are held to each other by no compiler "
                            "here, and a row short of a cell is a combination nobody decided")

    # ---- the per-class table ---------------------------------------------------------------
    print()
    print(f"{'device':7s} {'precision':14s} {'shape':15s} {'row':4s} {'states':7s} "
          f"{'marker':9s} {'budget':9s} {'axes':5s} {'statable':8s} verdict")
    print("-" * 108)

    row_by_class = {row.klass: row for row in rows}
    axes_ok = 0
    cell_axis_ok = 0
    for row in rows:
        if len(row.cells) < 10:
            failures.append(f"the row at {os.path.relpath(row.source, REPO)}:{row.line} carries "
                            f"{len(row.cells)} cell(s) and the format takes {len(cells_named)}")
            continue
        if bare(row.cells[0]) not in device_members:
            failures.append(f"the row at {row.line} names the device cell {row.cells[0]}, which is "
                            f"no member of Device ({', '.join(device_members)})")
        if bare(row.cells[1]) not in lane_members:
            failures.append(f"the row at {row.line} names the precision cell {row.cells[1]}, which "
                            f"is no member of Precision")
        if bare(row.cells[2]) not in shape_members:
            failures.append(f"the row at {row.line} names the shape cell {row.cells[2]}, which is "
                            f"no member of Shape")
        # One cell per axis, and the cell names a member of the axis it belongs to.
        seen: dict[int, str] = {}
        for cell in range(3, 10):
            owners = [kind for kind, members in axis_members.items()
                      if bare(row.cells[cell]) in members]
            if len(owners) != 1:
                failures.append(f"the row at {row.line} cell {cell} ({row.cells[cell]}) belongs to "
                                f"{len(owners)} axis/axes: a cell that names no member of one axis "
                                "is a combination the row cannot be read as")
                continue
            seen[cell] = owners[0]
        if len(seen) == 7 and len(set(seen.values())) == 7:
            cell_axis_ok += 1
            order = [axes[i][0] for i in range(len(axes))]
            got = [seen[cell] for cell in range(3, 10)]
            if got != order:
                failures.append(f"the row at {row.line} writes its axis cells as {got} and "
                                f"StatedEvalPolicy takes them as {order}")

    key_set = set(key)
    row_set = set(row_by_class)

    for device, precision, shape in key:
        row = row_by_class.get((device, precision, shape))
        if row is None:
            print(f"{device:7s} {precision:14s} {shape:15s} {'NO':4s} {'-':7s} {'-':9s} "
                  f"{'-':9s} {'0':5s} {'0':8s} NO DEFAULT: a consumer naming no policy gets "
                  "nothing here")
            failures.append(f"({device}, {precision}, {shape}) is a class the library reaches and "
                            "the seam carries no row for: a call that names no policy is answered "
                            "with a compile error, not with a choice")
            continue
        is_device = device == "kDevice"
        states, differing = row_states_seam_names(row, names, is_device, host_exp)
        kind = marker_kind(row.marker)
        statable = 1
        for _kind, members in axis_members.items():
            statable *= len(members)
        axes_ok += 1
        verdict = "capturable"
        if kind == "unknown":
            verdict = "capturable, marker not read"
            notes.append(f"the row for ({device}, {precision}, {shape}) carries a marker this "
                         f"reader does not classify: {row.marker[:80]}")
        print(f"{device:7s} {precision:14s} {shape:15s} {'yes':4s} "
              f"{'seam' if states else 'moved':7s} {kind:9s} {row.cells[5]:9s} "
              f"{'7/7':5s} {statable:<8d} {verdict}")
        if not states and kind != "measured":
            notes.append(f"({device}, {precision}, {shape}): the marker says {kind} and the cells "
                         f"are not the file's own names - {differing[0]}")

    if rows:
        print()
        print(f"  {axes_ok} of {len(rows)} row(s) name a member of every axis; "
              f"{cell_axis_ok} of {len(rows)} write the seven cells in the policy's own order")

    missing_rows = sorted(key_set - row_set)
    outside = sorted(row_set - key_set)
    for klass in outside:
        row = row_by_class[klass]
        failures.append(f"the seam carries a row for ({klass[0]}, {klass[1]}, {klass[2]}) at "
                        f"{os.path.relpath(row.source, REPO)}:{row.line} and the library reaches no "
                        "such class: a default nothing can ask for")

    # ---- the measurement's key against the seam's ------------------------------------------
    print()
    probe_cells = [str(cell) for cell in probe["cells"]]
    lane_of = probe["laneOf"]
    print(f"THE MEASUREMENT'S KEY, as src/boys_probe.cpp states it: kCellPrecisions = "
          f"{', '.join(probe_cells)}")
    print(f"  the seam's own shapes it names: {', '.join(str(s) for s in probe['seamShapes'])}; "
          f"the device questions: {', '.join(str(s) for s in probe['deviceShapes'])}")

    lane_of = {k: v for k, v in lane_of.items()} if isinstance(lane_of, dict) else {}
    folded_groups: dict[str, list[str]] = {}
    if not lane_of:
        failures.append("LaneOf could not be read from src/boys_probe.cpp: which class of the "
                        "measurement's key each seam lane answers for is not known here, and a "
                        "check that cannot read its own key must not pass")
    else:
        folded: dict[str, list[str]] = {}
        for cell in probe_cells:
            lane = lane_of.get(cell)
            if lane is None:
                failures.append(f"the probe's class {cell} maps to no lane of Precision: the two "
                                "keys cannot be compared")
                continue
            folded.setdefault(str(lane), []).append(cell)
        folded_groups = folded
        for lane, cell_list in sorted(folded.items()):
            if len(cell_list) > 1:
                own = [cell for cell in cell_list if cell == lane]
                others = [cell for cell in cell_list if cell != lane]
                print(f"  FOLD: {len(cell_list)} class(es) of the measurement - "
                      f"{', '.join(cell_list)} - share the one seam cell {lane}")
                shapes = list(probe["seamShapes"]) or ["<shape list not read>"]
                print(f"        the row of {lane} is written for {own[0] if own else cell_list[0]} "
                      f"(the lane's own name); {', '.join(others)} has no cell of its own, over "
                      f"{len(shapes)} shape(s)")
                failures.append(
                    f"the measurement keys {', '.join(others)} and the seam's key cannot name it: "
                    f"the class(es) resolve to the {lane} row, so a default measured for "
                    f"{', '.join(others)} cannot be captured - a format-specific default needs a "
                    "format cell that Precision does not have")
            else:
                print(f"  cell {cell_list[0]} -> {lane}")

    # ---- the device half: what a call naming no policy gets ---------------------------------
    form_name, exp_name, form_count, exp_count, policy_count = device_call_default()
    print()
    print("THE DEVICE HALF, what a device call that names no policy gets")
    print(f"  the seam's two device names: {names.get('deviceDivision')} and "
          f"{names.get('deviceExp')} (boys/boys_build_defaults.hpp)")
    print(f"  read off the lane's declarations ({os.path.relpath(CUDA_FILE, REPO)}): "
          f"{form_count} parameter default(s) = {form_name}, {exp_count} template default(s) = "
          f"{exp_name}, and {policy_count} policy parameter(s)")
    if policy_count == 0:
        print("  no device entry defaults a policy: the nine device rows are reached by naming "
              "DefaultPolicy<..., Device::kDevice> (and by DefaultGuarantee), not by an entry")
    device_rows = [row for row in rows if row.cells and row.cells[0] == "kDevice"]
    differing_forms = [row.klass for row in device_rows
                       if len(row.cells) > 8 and row.cells[8] != names.get("deviceDivision")]
    print(f"  {len(device_rows)} device row(s); {len(differing_forms)} of them state a division "
          f"form that is not the seam's device name: {', '.join(' '.join(k[1:]) for k in differing_forms)}")
    print("  so per class: the entry's own body (frozen in its name), the seam's device division "
          "form and the lane's region-B exponential - a CHOICE the build states for the lane, and "
          "not a per-class fallback. The row's other cells reach a caller only through "
          "DefaultPolicy<..., Device::kDevice> by name.")

    # ---- the half-format fold, against a run's report --------------------------------------
    print()
    if args.report:
        text = read(args.report)
        if not text:
            failures.append(f"the report {args.report} could not be read: the half-format fold is "
                            "not held to anything")
        else:
            found = re.findall(r"the two half formats' winners were not one combination in the "
                               r"(\S+) class", text)
            size = os.path.getsize(args.report)
            print(f"THE HALF-FORMAT FOLD, against {args.report} ({size} byte(s))")
            stating = text.splitlines()[0] if text.splitlines() else ""
            print(f"  the run states itself: {stating}")
            half_formats = [cell for group in folded_groups.values() if len(group) > 1
                            for cell in group]
            for klass in found:
                leaders = report_leaders(text, half_formats, klass)
                for fmt, (entry, how) in sorted(leaders.items()):
                    print(f"    {fmt} {kebab(klass)}: {entry} ({how})")
                combos = {combination_of(entry, half_formats)
                          for entry, _how in leaders.values()}
                if len(combos) > 1:
                    print("    the two are not one combination: one row of the seam cannot carry "
                          "both, and the lane's row is written for one of them")
                elif len(leaders) > 1:
                    print("    the two are one combination under their formats: this run's leaders "
                          "agree, and its verdict sentence above says otherwise only if it is "
                          "about another class")
            if found:
                print(f"  the report says the two winners DIFFER, in {len(found)} class(es): "
                      f"{', '.join(found)}")
                failures.append("the run reports the two half formats' winners were not one "
                                f"combination in the {', '.join(found)} class(es): the bf16 class's "
                                "winner cannot be the consumer's default, and the fp16 row is what "
                                "a bf16 call runs")
            elif half_formats:
                print("  the report carries no disagreement between the two half formats: the "
                      "probe prints that sentence only where they differ, so this run's bf16 "
                      "classes ran the fp16 class's winning combination")
            else:
                print("  the report carries no disagreement between the two half formats, and the "
                      "measurement's key here names no class the seam's key cannot: there is no "
                      "fold for a comparison to be about")
    else:
        half_rows = [row for row in rows if len(row.cells) > 1 and row.cells[1] == "kFp16"]
        claim = ""
        for row in half_rows:
            if "half formats" in row.marker:
                claim = row.marker
                break
        print("THE HALF-FORMAT FOLD, against the tree")
        print("  no --report: whether the two half classes' measured winners are one combination "
              "is a measurement, and the tree carries no run. What the tree carries is the claim:")
        if claim:
            print(f'    the seam\'s own row: "{claim[:100]}"')
        print("  it is a claim and not a transcript - pass --report with the run's output to hold "
              "it to the probe's own verdict")

    # ---- the verdict -----------------------------------------------------------------------
    print()
    sys.stdout.flush()          # so a piped run reads findings after the tables they are about
    if failures:
        for failure in failures:
            print(f"check_default_capturability: {failure}", file=sys.stderr)
    for note in notes:
        print(f"check_default_capturability: note: {note}", file=sys.stderr)

    if args.space:
        print()
        wanted = tuple(part.strip() for part in args.space.split(","))
        row = row_by_class.get(wanted)
        if row is None:
            print(f"check_default_capturability: no row for {wanted}", file=sys.stderr)
        else:
            print(f"the combination space of {wanted} as the row format can state it:")
            for axis_type, param in axes:
                print(f"  {param}: {', '.join(axis_members[axis_type])}")
            total = 1
            for _kind, members in axis_members.items():
                total *= len(members)
            print(f"  {total} combination(s), every one of them statable in a row of this format")

    if failures:
        print(f"\ncheck_default_capturability: {len(failures)} finding(s)",
              file=sys.stderr)
        print(f"check_default_capturability: FAIL - {len(failures)} finding(s), listed on stderr: "
              "a class the seam's key reaches, or a class the measurement keys, cannot be the "
              "consumer's default")
        return 1

    print(f"\ncheck_default_capturability: PASS - {len(key)} class(es) of the seam's key carry a "
          f"row, every one of the {len(axes)} axes has a cell, and every class the measurement "
          "keys has a cell of its own")
    return 0


if __name__ == "__main__":
    sys.exit(main())
