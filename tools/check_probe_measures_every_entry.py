#!/usr/bin/env python3
"""Every entry the public surface declares is an entry the option probe calls.

The probe measures the library's options and the report it writes is read as an account of
them, so an entry the surface declares and the probe never calls is a class whose row says
"no option of this precision and shape produced a figure on this run" - a sentence about the
probe that reads as a sentence about the library. That is how five entries went unmeasured:
`BoysFixedNF32`, `BoysAllNAtOrdersF32`, `BoysFixedNF16`, `BoysAllNF16` and
`BoysAllNAtOrdersF16` were declared, served and never timed, and the five seam rows for their
classes were written from the file's own names instead of from a measurement.

Both sides are read from source and neither is a list written here: the declared entries come
from the public header and the calls come from the probe's own translation unit.

An entry is a declaration, not a name. `BoysAllN` declares two entries that differ by their
last parameter: one takes a workspace, and one takes `BoysSortedArgs`, which is a caller
promising its arguments arrive already non-decreasing. A reading keyed on the name sees one
entry where the surface declares two, and the probe could stop calling one of them and still
read as measuring the name - which is what happened: when the probe's nine shape rows became a
per-class space, the sorted-argument call left the space and no check could tell, because the
name `BoysAllN` was still called. The value-returning single-order entries (`BoysSingle`,
`BoysSingleF32`, `BoysSingleF16`, `BoysSingleBf16`) were missed by the same reading's `void`
and are counted here with everything else.

The declared side is the declarations at the top level of `namespace boys` in the one public
header that declares the host entries this probe measures. The span and C spellings of the same
shapes live in other headers of this library, and `tools/check_host_entry_bijection.py` is the
check that holds every declared name to a shape; `boys::detail` is implementation and not
surface. The called side is the probe's own translation unit: a library translation unit's
definition of an entry is a definition and not a call, and reading one as a call was how
`BoysAllNF16Native` - declared, defined and never called by the probe - read as measured. A
`requires`-expression's body is a requirement rather than a call the probe makes, and the
probe's capability traits are full of them.

An entry the probe may not measure is named in EXCLUDED with the reason, and a reason has to
be a fact about the question rather than about the work: an entry that answers a different
question is not the same entry under another name, and one that is merely expensive is work.

usage: check_probe_measures_every_entry.py [--root DIR]
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import re
import sys

# The one header that declares the host entries this probe measures, and the sources a probe
# measures from. The CUDA surface is a separate build and a separate probe
# (`boys-device-probe`), so it is not read here; a host entry is what this check is about.
DECLARING_HEADERS = ["include/boys/boys.hpp"]
PROBE_SOURCES = ["src/boys_probe.cpp"]

# What makes a declaration an entry: it returns `void` and writes its results, or it returns
# one floating figure; and it takes the order it evaluates - `int n`, the highest order
# `int nmax`, or the order array `const int* n` - at a floating argument. That is the question
# the probe times. The declarations beside the entries answer questions of their own
# (`BoysAccuracyGuaranteed`, `BoysAllNWorkspaceSize`, `BoysEvalSchemeDelivered`) and they are
# printed with the entries, so that no declaration this read passes over is passed over
# silently.
ENTRY_RETURNS = frozenset({"void", "double", "float", "F16", "Bf16", "Half2"})
FLOATING_TYPES = ("double", "float", "F16", "Bf16", "Half2")
ORDER_PARAMETER = re.compile(r"^(?:const\s+)?int\s+(?:n|nmax)$|^const\s+int\s*\*\s*n$")

# Entries the probe may not measure, each keyed by the declaration's own signature - name and
# parameter types - with the fact about the question that excludes it. Keyed by the signature
# and not by the name, because a name is what this check stopped keying on.
EXCLUDED = {
    "BoysAllNF16Native(int, const F16*, F16*, std::size_t)": (
        "the native half lane: region C alone, orders 0..8, and its results are scaled by a "
        "power of two. That is a different question from the workload the probe times, whose "
        "rows are read as F_n of the argument they were handed, and a row of it would be a "
        "figure for another question under this one's name."
    ),
    "BoysAllOrdersHalf2(int, Half2, Half2*)": (
        "the same native half lane taken packed: region C alone, orders 0..8, and its packed "
        "output is scaled by 2^15. It is the type's own entry rather than the probe's question "
        "for the same reason as `BoysAllNF16Native` above - the figure would be another "
        "question's under this one's name."
    ),
}

NAME = r"Boys[A-Za-z0-9_]*"
TAG_STRUCT = re.compile(r"\bstruct\s+(" + NAME + r")\s*\{\s*\}\s*;")
KEYWORDS = frozenset({"constexpr", "inline", "static", "extern", "const", "consteval"})


def blank(text: str, start: int, end: int) -> str:
    """The text with [start, end) replaced by spaces, newlines kept in place."""

    return text[:start] + "".join("\n" if c == "\n" else " " for c in text[start:end]) + text[end:]


def stripped(text: str) -> str:
    """The text with comments, string literals and `requires`-bodies blanked out.

    A declaration or a call written in a comment is prose, a quoted one is a string, and a
    `requires`-expression's body is a requirement: none of the three is an entry the probe
    evaluates, and this file's own prose quotes declarations. A trailing `requires` clause -
    `requires Foo<T> { ... }`, a lambda's - is not blanked, because the body after it is a
    body; only the parenthesised form, which is what a requires-*expression* has, is.
    """

    out = stripped_comments(text)

    for match in re.finditer(r"\brequires\b", out):
        i = match.end()

        while i < len(out) and out[i].isspace():
            i += 1

        if i < len(out) and out[i] == "(":
            close = matching(out, i, "(", ")")
            if close < 0:
                continue

            j = close + 1

            while j < len(out) and out[j].isspace():
                j += 1

            if j < len(out) and out[j] == "{":
                end = matching(out, j, "{", "}")
                if end < 0:
                    continue

                out = blank(out, j, end + 1)

    return out


def stripped_comments(text: str) -> str:
    """The text with comments and string and character literals blanked out."""

    return _scan_literals(text)


def _scan_literals(text: str) -> str:
    pieces: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        two = text[i : i + 2]

        if two == "//":
            j = text.find("\n", i)
            j = n if j < 0 else j
            pieces.append(blank(text[i:j], 0, j - i))
            i = j
        elif two == "/*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            pieces.append(blank(text[i:j], 0, j - i))
            i = j
        elif text[i] in "\"'":
            quote = text[i]
            j = i + 1

            while j < n and text[j] != quote:
                j += 2 if text[j] == "\\" else 1

            j = min(j + 1, n)
            pieces.append(blank(text[i:j], 0, j - i))
            i = j
        else:
            pieces.append(text[i])
            i += 1

    return "".join(pieces)


def matching(text: str, start: int, opener: str, closer: str) -> int:
    """The index of the `closer` matching the `opener` at `start`, or -1."""

    depth = 0
    i = start

    while i < len(text):
        if text[i] == opener:
            depth += 1
        elif text[i] == closer:
            depth -= 1
            if depth == 0:
                return i
        i += 1

    return -1


def split_arguments(text: str) -> list[str]:
    """The text's top-level comma-separated arguments, angle brackets included."""

    arguments: list[str] = []
    depth = 0
    current: list[str] = []

    for ch in text:
        if ch in "(<[{":
            depth += 1
        elif ch in ")>]}":
            depth -= 1

        if ch == "," and depth == 0:
            arguments.append("".join(current))
            current = []
        else:
            current.append(ch)

    tail = "".join(current)
    if tail.strip() or arguments:
        arguments.append(tail)

    return [argument.strip() for argument in arguments if argument.strip()]


@dataclasses.dataclass(frozen=True)
class Parameter:
    """One declared parameter: its type, whether it carries a default, and its tag if any."""

    type: str
    text: str
    defaulted: bool
    tag: bool


@dataclasses.dataclass(frozen=True)
class Entry:
    """One declared entry, identified by its declaration rather than by its name."""

    name: str
    parameters: tuple[Parameter, ...]
    path: str
    line: int

    @property
    def signature(self) -> str:
        return f"{self.name}({', '.join(p.type for p in self.parameters)})"

    @property
    def required(self) -> int:
        """How many arguments a call must pass: the parameters without a default."""

        return sum(1 for p in self.parameters if not p.defaulted)


@dataclasses.dataclass(frozen=True)
class Call:
    """One call of a `Boys*` name in a probe source, with its argument texts."""

    name: str
    arguments: tuple[str, ...]
    path: str
    line: int


def parameter_of(text: str, tags: frozenset[str]) -> Parameter:
    """Parse one declared parameter."""

    defaulted = "=" in text
    body = text.split("=", 1)[0].strip()
    words = body.split()
    type_text = body

    # A parameter's name is the last word when there is more than one: `const double* x` is
    # `const double*`. One word alone is the type - `BoysSortedArgs` is declared unnamed - and
    # `double*` keeps its star, so a lone word is never taken for a name.
    if len(words) > 1 and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", words[-1]):
        type_text = " ".join(words[:-1])

    if not type_text:
        type_text = body

    return Parameter(type=type_text, text=body, defaulted=defaulted, tag=type_text in tags)


def brace_spans(text: str) -> list[tuple[int, int, int | None]]:
    """Every brace pair as (open, close, parent), parents being the innermost containing pair."""

    spans: list[tuple[int, int, int | None]] = []
    stack: list[int] = []

    for i, ch in enumerate(text):
        if ch == "{":
            parent = stack[-1] if stack else None
            spans.append([i, -1, parent])  # type: ignore[arg-type]
            stack.append(len(spans) - 1)
        elif ch == "}" and stack:
            index = stack.pop()
            spans[index][1] = i

    return [(open_, close, parent) for open_, close, parent in spans]


def surface_regions(text: str) -> list[tuple[int, int, list[tuple[int, int]]]]:
    """The top-level `namespace boys` ranges, each with the child ranges to leave out.

    A declaration that sits inside a nested namespace, a struct or a function body is
    implementation, not surface: only what sits directly in `namespace boys` is read.
    """

    spans = brace_spans(text)
    regions: list[tuple[int, int, list[tuple[int, int]]]] = []

    for index, (open_, close, parent) in enumerate(spans):
        if parent is not None or close < 0:
            continue

        header = text[max(0, open_ - 80) : open_]
        if not re.search(r"namespace\s+boys\s*$", header):
            continue

        children = [(child_open, child_close) for i, (child_open, child_close, p) in enumerate(spans)
                    if p == index and child_close > 0]
        regions.append((open_, close, children))

    return regions


def inside(ranges: list[tuple[int, int]], position: int) -> bool:
    return any(open_ < position < close for open_, close in ranges)


def declarations(text: str, path: str, tags: frozenset[str]) -> list[tuple[Entry, bool]]:
    """Every `Boys*` declaration at namespace scope, with whether it is an entry.

    The return type is what is written before the name - a query beside the entries returns
    its answer as a bool, a count or a record - and the entry rule is the one stated at the
    top of this file.
    """

    found: list[tuple[Entry, bool]] = []

    for region_open, region_close, children in surface_regions(text):
        body = text[region_open : region_close + 1]

        for match in re.finditer(r"\b(" + NAME + r")\s*\(", body):
            start = region_open + match.start()
            # The child ranges are positions in the whole text, as `start` is.
            if inside(children, start):
                continue

            name = match.group(1)
            head = text[last_boundary(text, region_open, start) : start]
            head = strip_template_clause(head).strip()
            return_type = head.split()[-1] if head.split() else ""

            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_:<>,*&\s]*", return_type):
                continue

            open_paren = text.find("(", start + len(name))
            close = matching(text, open_paren, "(", ")")
            if close < 0:
                continue

            parameters = tuple(parameter_of(part, tags)
                               for part in split_arguments(text[open_paren + 1 : close]))

            entry = Entry(name=name, parameters=parameters, path=path,
                          line=text.count("\n", 0, start) + 1)

            is_entry = (
                return_type in ENTRY_RETURNS
                and any(ORDER_PARAMETER.fullmatch(p.text) for p in parameters)
                and any(floating(p.type) for p in parameters)
            )

            found.append((entry, is_entry))

    return found


def last_boundary(text: str, start: int, end: int) -> int:
    """The index just past the last `;`, `{` or `}` before `end`, not before `start`."""

    found = start
    for index in range(start, end):
        if text[index] in ";{}":
            found = index + 1

    return found


def strip_template_clause(head: str) -> str:
    """`head` without a leading `template <...>` clause."""

    stripped_head = head.lstrip()
    if not stripped_head.startswith("template"):
        return head

    open_angle = stripped_head.find("<")
    if open_angle < 0:
        return head

    depth = 0
    for index in range(open_angle, len(stripped_head)):
        if stripped_head[index] == "<":
            depth += 1
        elif stripped_head[index] == ">":
            depth -= 1
            if depth == 0:
                return stripped_head[index + 1 :]

    return head


def wrapped(items: list[str], width: int = 96) -> str:
    """The items as an indented, comma-separated, wrapped list."""

    lines: list[str] = []
    current = "  "

    for index, item in enumerate(items):
        text = item + ("," if index + 1 < len(items) else "")

        if len(current) + len(text) + 1 > width and current.strip():
            lines.append(current.rstrip())
            current = "  "

        current += text + " "

    if current.strip():
        lines.append(current.rstrip())

    return "\n".join(lines)


def floating(type_text: str) -> bool:
    return any(re.search(r"\b" + re.escape(t) + r"\b", type_text) for t in FLOATING_TYPES)


def tag_types(text: str) -> frozenset[str]:
    """The tag types the header declares: an empty struct is what a tag is."""

    return frozenset(TAG_STRUCT.findall(text))


def calls(text: str, path: str, tags: frozenset[str]) -> list[Call]:
    """Every call of a `Boys*` name in a probe source, template arguments and all."""

    found: list[Call] = []
    source = stripped(text)

    for match in re.finditer(r"\b(" + NAME + r")\b", source):
        name = match.group(1)
        i = match.end()

        while i < len(source) and source[i].isspace():
            i += 1

        if i < len(source) and source[i] == "<":
            close_angle = matching(source, i, "<", ">")
            if close_angle < 0:
                continue

            i = close_angle + 1

            while i < len(source) and source[i].isspace():
                i += 1

        if i >= len(source) or source[i] != "(":
            continue

        close = matching(source, i, "(", ")")
        if close < 0:
            continue

        found.append(Call(name=name, arguments=tuple(split_arguments(source[i + 1 : close])),
                          path=path, line=source.count("\n", 0, match.start()) + 1))

    return found


def accepts(entry: Entry, call: Call, tags: frozenset[str]) -> bool:
    """Whether this call is a call of this declaration.

    A declaration that takes a tag is called with the tag spelled out and a declaration that
    takes none is not: that is the difference between `BoysAllN`'s two declarations, and the
    one an arity-only reading cannot see.
    """

    if entry.name != call.name or not (entry.required <= len(call.arguments) <= len(entry.parameters)):
        return False

    for index, argument in enumerate(call.arguments):
        parameter = entry.parameters[index]
        named = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:\{\s*\}|\(\s*\))", argument)
        argument_tag = named.group(1) in tags if named else False

        if parameter.tag != argument_tag:
            return False

        if argument_tag and named is not None and named.group(1) != parameter.type:
            return False

    return True


def read(root: pathlib.Path) -> tuple[list[tuple[Entry, bool]], list[Call], frozenset[str]]:
    declared: list[tuple[Entry, bool]] = []
    called: list[Call] = []
    tags: frozenset[str] = frozenset()

    for relative in DECLARING_HEADERS:
        path = root / relative
        if not path.exists():
            print(f"check_probe_measures_every_entry: {relative} is not in this tree: the "
                  "declared side of this check names a header that does not exist")
            return [], [], frozenset()

        text = stripped(path.read_text(encoding="utf-8", errors="replace"))
        tags = tags | tag_types(text)
        declared.extend(declarations(text, relative, tags))

    for relative in PROBE_SOURCES:
        path = root / relative
        if not path.exists():
            print(f"check_probe_measures_every_entry: {relative} is not in this tree: the "
                  "called side of this check names a source that does not exist")
            return [], [], frozenset()

        called.extend(calls(path.read_text(encoding="utf-8", errors="replace"), relative, tags))

    return declared, called, tags


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="the repository root")
    arguments = parser.parse_args()
    root = pathlib.Path(arguments.root).resolve()

    declared, called, tags = read(root)

    if not declared:
        print("check_probe_measures_every_entry: no entry declared - the read found nothing, "
              "which is a defect of this check and not a surface with no entry in it")
        return 1

    entries = [entry for entry, is_entry in declared if is_entry]
    beside = [entry for entry, is_entry in declared if not is_entry]

    if not entries:
        print("check_probe_measures_every_entry: the read found declarations and no entry "
              "among them - a defect of this check and not a surface with no entry in it")
        return 1

    measured = [entry for entry in entries
                if any(accepts(entry, call, tags) for call in called)]
    excluded = [entry for entry in entries if entry.signature in EXCLUDED]
    missed = [entry for entry in entries if entry not in measured and entry not in excluded]
    unmatched = [call for call in called
                 if call.name in {entry.name for entry in entries}
                 and not any(accepts(entry, call, tags) for entry in entries)]

    names = {entry.name for entry in entries}
    print(f"declared entries, by declaration: {len(entries)} "
          f"({len(names)} name(s), {len(entries) - len(names)} of them sharing a name with "
          "another declaration)")
    print(f"entries the probe calls: {len(measured)}")
    print(f"excluded with a reason: {len(excluded)}")

    for entry in excluded:
        print(f"  - {entry.signature}: {EXCLUDED[entry.signature]}")

    # The overloads are printed one declaration each: a name-level reading is what let one of
    # these drop out of the probe unnoticed, and this is the line that shows it cannot.
    overloaded = [name for name in sorted(names)
                  if sum(1 for entry in entries if entry.name == name) > 1]

    if overloaded:
        print(f"names carrying more than one declaration: {len(overloaded)}")
        for name in overloaded:
            for entry in entries:
                if entry.name != name:
                    continue

                print(f"  {entry.signature}: "
                      f"{'called' if entry in measured else 'excluded' if entry in excluded else 'never called'}")

    if missed:
        print(f"declared and never called by the probe: {len(missed)}")
        for entry in missed:
            print(f"  * {entry.signature}  ({entry.path}:{entry.line})")

    if unmatched:
        print(f"a call of an entry name that no declaration answers: {len(unmatched)}")
        for call in unmatched:
            print(f"  * {call.name}({', '.join(call.arguments)})  ({call.path}:{call.line})")

    # The declarations the entry rule passes over are printed too: a check that reads a header
    # and says nothing about most of what it read is a check whose reading cannot be audited.
    if beside:
        print(f"declared beside the entries (not entries, and so not this check's subject): "
              f"{len(beside)}")
        print(wrapped([entry.signature for entry in beside]))

    if missed or unmatched:
        print("A class whose only entry is one of these has no figure in the report and its seam "
              "row is written from the file's own names - the probe's account of the library is "
              "short by exactly these entries.")
        return 1

    print("every declared entry is an entry this probe calls")
    return 0


if __name__ == "__main__":
    sys.exit(main())
