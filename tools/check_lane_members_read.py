#!/usr/bin/env python3
r"""Check that every member a lane struct declares is one a body reads.

`src/boys_cuda.cu` declares the lane structs the device bodies are parameterised
over. A body reads a lane through its members: `DeviceSeed` reads `Count`, `A`,
`B`, `Coeffs` and `Deg`, and the region-B seat reads `BSeed` and nothing else. A
lane that a call site only ever places in the region-B seat therefore carries a
region-A half nothing reads, and nothing in this tree said so until this check.

What makes that a defect rather than tidiness is the direction the interface
fails in. The bodies of this library read a lane's region-A half by name and
unconditionally, so a lane whose region-A half is *missing* is a compile error
naming the member - which is the good direction, and it is why that half is
stated where a body reads it and nowhere else.

One name is not in that half. `kMonomial` is read by the `LaneMonomial` trait
and by nothing else, and that trait is a detection idiom: a name it does not find
is not an error, it is the Chebyshev answer. A lane's stored form is the lane's
own statement about the tables it was handed (boys_cuda_arithmetic.hpp), and the
entry table names the float monomial lanes as stating it (boys_cuda_options.hpp).
So the name's removal is the one removal here that is neither compile-checked nor
reported by the compiler as anything but an absence, and this check holds it
rather than counting it as a member a body reads.

Both sides are read off the tree rather than written down here:

  * the lane structs and their members, from the struct definitions in the
    source;
  * the bodies that read a lane, and the member each body reads from each of its
    lane parameters, from the bodies' own parameter lists and bodies;
  * which lane reaches which parameter, from the call sites and from the
    template arguments the kernels are launched with - a fixpoint, because a
    body may hand its own lane parameter to another body;
  * the names whose absence a body tolerates, from the detection idioms the
    headers spell. A `void_t<decltype(T::M)>` is a name a body asks about and
    does without, which is precisely the set this check must not report.

The scope is the source this script is pointed at, and it says so rather than
implying more. `include/boys/boys_cuda_device.hpp` declares a second set of lane
structs - the device-callable ones, whose tables are handed in - and they are not
read here: they carry a plain data member and are class templates, and this
script's struct reader refuses both rather than skipping them, so pointing
`--source` at that header stops the run instead of passing it. The compiler's own
`#177-D` is a statement about one translation unit, and the one this check reads -
`src/boys_cuda.cu` - does not include that header, so the 17 members nvcc names
there are that file's share and not the tree's.

Usage:

    python3 tools/check_lane_members_read.py --check
    python3 tools/check_lane_members_read.py --check --source src/boys_cuda.cu

Exit status: 0 when every lane member is read by a body or held as an interface
member; 1 when a member is neither, each named with its lane and the line it is
declared on; 2 when the input cannot be read - a struct or a body this script
does not understand is reported rather than skipped, because a parse that
quietly matches less is a check that quietly passes.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parent.parent

# The source that carries the lane structs and the kernels, and the header that
# carries the bodies. Both are named here as the two files this check reads; the
# structs and the bodies inside them are what the run derives.
SOURCE = REPO / "src" / "boys_cuda.cu"
ARITHMETIC = REPO / "include" / "boys" / "boys_cuda_arithmetic.hpp"

# Where a detection idiom may sit. Scanned for the `void_t<decltype(T::M)>` form,
# so a name a body asks about and does without is read off the headers.
HEADERS = REPO / "include" / "boys"

# A struct definition at the head of a line, and its closing brace, which the
# style puts in the first column.
STRUCT_OPEN = re.compile(r"^struct (\w+) \{\s*$")
STRUCT_CLOSE = re.compile(r"^\};")

# The two member forms a lane declares: a device method, and a static constant.
# The name is the identifier before the parameter list, and before the `=`.
METHOD = re.compile(r"^\s*__device__\s+__forceinline__\s+.*?\b(\w+)\s*\(")
CONSTANT = re.compile(r"^\s*(?:static|inline|constexpr)[^=;]*?\b(\w+)\s*=")

# A function template, and the device marker its signature must carry for this
# script to read the parameter list that follows. A template whose signature
# carries neither marker is a class template and is not a body.
TEMPLATE = re.compile(r"^template\s*<", re.MULTILINE)
MARKERS = ("__device__", "__global__")
FUNCTION = re.compile(r"\b(\w+)\s*\((?=[^()]*(?:\([^()]*\)[^()]*)*\s*$)")

# A generic parameter of the enclosing template, and a lane-typed function
# parameter: `typename T` names a type, and `const T& t` is the seat a lane is
# passed into.
TYPE_PARAMETER = re.compile(r"^typename\s+(\w+)$")
LANE_PARAMETER = re.compile(r"^\s*const\s+([A-Za-z_]\w*)\s*&\s*([A-Za-z_]\w*)\s*$")

# A read of a member off a name.
MEMBER_READ = re.compile(r"\b(\w+)\s*\.\s*(\w+)")

# A call, with an optional template argument list one level deep.
CALL = re.compile(r"\b(\w+)\s*(<[^<>]*(?:<[^<>]*>[^<>]*)*>)?\s*\(")

# The detection idiom: a name a body asks about and does without.
DETECTION = re.compile(r"void_t\s*<\s*decltype\s*\(\s*\w+\s*::\s*(\w+)\s*\)")

# A name handed as a value or a template argument: `Lane64Full{}` or `Lane64Full`.
LANE_ARGUMENT = re.compile(r"^([A-Za-z_]\w*)\s*\{?\s*\}?$")


class ScriptError(Exception):
    """A reading this script does not understand. Reported, never skipped."""


class Signature(NamedTuple):
    """A function template: its name, its template header, its body's extent."""

    name: str
    header: str
    parameters: str
    start: int
    end: int


def strip_comments(text: str) -> str:
    """The text with its comments blanked, at their own length so lines keep theirs."""
    out = []
    index = 0
    length = len(text)

    while index < length:
        two = text[index : index + 2]

        if two == "//":
            end = text.find("\n", index)
            end = length if end < 0 else end
            out.append(" " * (end - index))
            index = end
        elif two == "/*":
            end = text.find("*/", index + 2)
            end = length if end < 0 else end + 2
            out.append("".join("\n" if c == "\n" else " " for c in text[index:end]))
            index = end
        else:
            out.append(text[index])
            index += 1

    return "".join(out)


def balanced(text: str, start: int, opening: str, closing: str) -> int:
    """The index just past the group that opens at `start`. Raises when it never closes."""
    depth = 0

    for index in range(start, len(text)):
        if text[index] == opening:
            depth += 1
        elif text[index] == closing:
            depth -= 1

            if depth == 0:
                return index + 1

    raise ScriptError(f"an unclosed {opening!r} at offset {start}")


def split_top_level(text: str) -> list[str]:
    """The comma-separated items of an argument or parameter list."""
    items = []
    depth = 0
    current = []

    for character in text:
        if character in "(<[{":
            depth += 1
        elif character in ")>]}":
            depth -= 1

        if character == "," and depth == 0:
            items.append("".join(current).strip())
            current = []
        else:
            current.append(character)

    tail = "".join(current).strip()

    if tail:
        items.append(tail)

    return items


def signatures(text: str) -> list[Signature]:
    """Every function template this script reads, with its generic parameter names."""
    found = []

    for match in TEMPLATE.finditer(text):
        header_start = match.end() - 1
        header_end = balanced(text, header_start, "<", ">")
        header = text[header_start + 1 : header_end - 1]

        stop = min(
            (position for position in (text.find("{", header_end), text.find(";", header_end)) if position >= 0),
            default=-1,
        )

        if stop < 0:
            raise ScriptError(
                f"a template at offset {match.start()} declares neither a body nor an end"
            )

        if not any(marker in text[header_end:stop] for marker in MARKERS):
            continue

        opening = text.find("(", header_end)

        if opening < 0 or opening > stop:
            raise ScriptError(f"a device template at offset {match.start()} has no parameter list")

        parameters_end = balanced(text, opening, "(", ")")
        brace = text.find("{", parameters_end)

        if brace < 0 or brace > stop:
            raise ScriptError(f"a device template at offset {match.start()} declares no body")

        name = FUNCTION.search(text, header_end, opening + 1)

        if name is None:
            raise ScriptError(f"a device template at offset {match.start()} has no name")

        found.append(
            Signature(
                name.group(1),
                header,
                text[opening + 1 : parameters_end - 1],
                brace,
                balanced(text, brace, "{", "}"),
            )
        )

    return found


def template_parameters(header: str) -> list[tuple[str, bool]]:
    """A template header's parameters in order, each with whether it names a type.

    The order is what an explicit template argument list is read against, and the
    mark is what says which of those arguments can be a lane: a `DivisionForm` or
    a `bool` argument is a value, and reading one as a lane would put a name in the
    seats that no struct declares.
    """
    parameters = []

    for item in split_top_level(header):
        item = item.split("=", 1)[0].strip()
        generic = TYPE_PARAMETER.match(item)

        if generic:
            parameters.append((generic.group(1), True))
            continue

        named = re.search(r"\b(\w+)\s*$", item)

        if named is None:
            raise ScriptError(f"a template parameter this script cannot read: {item!r}")

        parameters.append((named.group(1), False))

    return parameters


class Lane:
    """A lane struct: its members, and the line each is declared on."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.members: dict[str, int] = {}


def lane_structs(text: str) -> dict[str, Lane]:
    """Every struct a struct definition at the head of a line declares, and its members.

    The members sit at one brace of depth, so the lines of a method's own body are
    read as the method's and not as a further declaration.
    """
    lanes: dict[str, Lane] = {}
    current: Lane | None = None
    depth = 0

    for number, line in enumerate(text.splitlines(), start=1):
        opened = STRUCT_OPEN.match(line)

        if opened:
            if current is not None:
                raise ScriptError(f"a struct opens inside {current.name}, at line {number}")

            current = Lane(opened.group(1))
            depth = 1
            continue

        if current is None:
            continue

        if depth == 1 and STRUCT_CLOSE.match(line):
            lanes[current.name] = current
            current = None
            continue

        if depth == 1:
            for pattern in (METHOD, CONSTANT):
                declared = pattern.match(line)

                if declared:
                    current.members.setdefault(declared.group(1), number)
                    break
            else:
                if line.strip():
                    raise ScriptError(
                        f"{current.name} declares something this script cannot read, "
                        f"at line {number}: {line.strip()[:70]!r}"
                    )

        depth += line.count("{") - line.count("}")

        if depth < 1:
            raise ScriptError(f"{current.name} closes at line {number} without its brace")

    if current is not None:
        raise ScriptError(f"the struct {current.name} never closes")

    return lanes


class Body:
    """A function template: the seats a lane can reach, and the members it reads."""

    def __init__(self, signature: Signature, origin: str) -> None:
        self.name = signature.name
        self.template_parameters = template_parameters(signature.header)
        self.type_parameters = [
            parameter for parameter, is_type in self.template_parameters if is_type
        ]
        self.origin = origin
        self.start = signature.start
        self.end = signature.end
        # A seat is a name a call site can hand a lane in under. It is a lane
        # parameter of this function, or a generic parameter of its template.
        # `roles` maps the name to the generic parameter the seat's type is, or
        # to None when the seat is a lane the caller names outright.
        self.roles: dict[str, str | None] = {}
        # The seat each function parameter is, in declaration order, so that an
        # argument list lines up with the seats it fills.
        self.positions: list[str | None] = []

        for item in split_top_level(signature.parameters):
            seat = LANE_PARAMETER.match(item)

            if seat is None:
                self.positions.append(None)
                continue

            self.positions.append(seat.group(2))
            self.roles[seat.group(2)] = seat.group(1) if seat.group(1) in self.type_parameters else None

        for parameter in self.type_parameters:
            self.roles.setdefault(parameter, None)

        # seat name -> the members read off it
        self.reads: dict[str, set[str]] = {seat: set() for seat in self.roles}

        for reader, member in MEMBER_READ.findall(origin[self.start : self.end]):
            if reader in self.reads:
                self.reads[reader].add(member)

    @property
    def takes_lanes(self) -> bool:
        """Whether a lane can reach this body at all."""
        return any(role is not None for role in self.roles.values()) or bool(self.type_parameters)


def bodies(text: str, origin: str) -> dict[str, Body]:
    """Every function template this script reads, whether or not it takes a lane."""
    return {signature.name: Body(signature, origin) for signature in signatures(text)}


def calls(text: str, known: set[str]) -> list[tuple[str, list[str]]]:
    """Every call to a known body, as (name, argument list)."""
    found = []

    for match in CALL.finditer(text):
        if match.group(1) not in known:
            continue

        closing = balanced(text, match.end() - 1, "(", ")")
        found.append((match.group(1), split_top_level(text[match.end() : closing - 1])))

    return found


def bindings(
    origin: str, generic: dict[str, Body], lanes: dict[str, Lane]
) -> dict[tuple[str, str], set[str]]:
    """The lane each generic parameter of each body is instantiated with.

    A kernel is instantiated by its launch, which names the lanes in its template
    argument list. A body instantiated by a call is reached through the fixpoint
    rather than here. An argument is read against the parameter it stands for, so a
    value argument is never taken for a lane.
    """
    bound: dict[tuple[str, str], set[str]] = {}

    for name, body in generic.items():
        for match in re.finditer(rf"\b{re.escape(name)}\s*<", origin):
            opening = match.end() - 1

            try:
                closing = balanced(origin, opening, "<", ">")
            except ScriptError:
                continue

            arguments = split_top_level(origin[opening + 1 : closing - 1])

            for index, (parameter, is_type) in enumerate(body.template_parameters):
                if not is_type or index >= len(arguments):
                    continue

                seat = LANE_ARGUMENT.match(arguments[index])

                if seat and seat.group(1) in lanes:
                    bound.setdefault((name, parameter), set()).add(seat.group(1))

    return bound


def reaches(
    contexts: dict[str, Body], generic: dict[str, Body], lanes: dict[str, Lane]
) -> dict[tuple[str, str], set[str]]:
    """Which lane reaches which seat of which body.

    A lane is named outright at some call sites and arrives at others through a
    body that hands its own seat on, so this is the fixpoint of the two: seeded
    where a call site names a lane and where a kernel's template argument list
    names one, and propagated along the arguments that name a seat.
    """
    known = set(generic)
    flows: dict[tuple[str, str], set[str]] = {}
    worklist: list[tuple[str, str]] = []

    def hand(target: Body, seat: str, arriving: set[str]) -> None:
        """Give `arriving` to a seat, and queue it when that is news."""
        if not arriving:
            return

        flow = flows.setdefault((target.name, seat), set())
        fresh = arriving - flow

        if fresh:
            flow.update(fresh)
            worklist.append((target.name, seat))

    for context in contexts.values():
        for callee, arguments in calls(context.origin[context.start : context.end], known):
            target = generic[callee]

            for index, argument in enumerate(arguments):
                if index >= len(target.positions):
                    break

                seat = target.positions[index]

                if seat is None:
                    continue

                named = LANE_ARGUMENT.match(argument)

                if named is not None and named.group(1) in lanes:
                    hand(target, seat, {named.group(1)})

    for origin in {body.origin for body in contexts.values()}:
        for (name, parameter), arriving in bindings(origin, generic, lanes).items():
            hand(generic[name], parameter, arriving)

    while worklist:
        body_name, seat = worklist.pop()
        context = contexts.get(body_name)
        arriving = flows.get((body_name, seat))

        if context is None or not arriving:
            continue

        for callee, arguments in calls(context.origin[context.start : context.end], known):
            target = generic[callee]

            for index, argument in enumerate(arguments):
                if index >= len(target.positions):
                    break

                target_seat = target.positions[index]

                if target_seat is None:
                    continue

                named = LANE_ARGUMENT.match(argument)

                if named is not None and named.group(1) == seat:
                    hand(target, target_seat, arriving)

    return flows


def interface_names(directory: Path) -> set[str]:
    """The member names a detection idiom asks about, and so does without."""
    names: set[str] = set()

    for path in sorted(directory.glob("*.hpp")):
        names.update(DETECTION.findall(path.read_text(encoding="utf-8")))

    return names


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that every lane member is read by a body.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="the checking mode: exit non-zero when a lane member has no reader",
    )
    parser.add_argument(
        "--source",
        default=str(SOURCE),
        help="the source carrying the lane structs (default: src/boys_cuda.cu)",
    )
    parser.add_argument(
        "--arithmetic",
        default=str(ARITHMETIC),
        help="the header carrying the bodies (default: include/boys/boys_cuda_arithmetic.hpp)",
    )
    parser.add_argument(
        "--headers",
        default=str(HEADERS),
        help="the directory scanned for detection idioms (default: include/boys)",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    options = parse_arguments(argv)

    if not options.check:
        print(
            "check_lane_members_read: no mode asked for; pass --check to make this a gate",
            file=sys.stderr,
        )
        return 0

    source_path = Path(options.source)
    arithmetic_path = Path(options.arithmetic)

    try:
        raw = {
            "source": source_path.read_text(encoding="utf-8"),
            "arithmetic": arithmetic_path.read_text(encoding="utf-8"),
        }
    except OSError as error:
        print(f"check_lane_members_read: {error}", file=sys.stderr)
        return 2

    stripped = {where: strip_comments(text) for where, text in raw.items()}

    try:
        lanes = lane_structs(stripped["source"])

        if not lanes:
            raise ScriptError(f"{source_path} declares no lane struct")

        contexts: dict[str, Body] = {}
        generic: dict[str, Body] = {}

        for where in ("source", "arithmetic"):
            for body in bodies(stripped[where], stripped[where]).values():
                contexts.setdefault(body.name, body)

                if body.takes_lanes:
                    generic.setdefault(body.name, body)

        flows = reaches(contexts, generic, lanes)
        placed = {lane for arriving in flows.values() for lane in arriving}

        if not placed:
            raise ScriptError("no lane reaches any seat - the call sites were not read")

        reads: dict[str, set[str]] = {name: set() for name in lanes}

        for (body_name, seat), arriving in flows.items():
            for lane in arriving:
                if lane in reads:
                    reads[lane].update(generic[body_name].reads.get(seat, set()))

        held = interface_names(Path(options.headers))
    except ScriptError as error:
        print(f"check_lane_members_read: {error}", file=sys.stderr)
        return 2

    findings: list[str] = []

    for name, lane in lanes.items():
        for member, line in sorted(lane.members.items(), key=lambda item: item[1]):
            if member in reads[name] or member in held:
                continue

            findings.append(
                f"{name}::{member} (line {line}) is declared, no body reads it, and no "
                f"detection idiom asks for it"
            )

    for name in sorted(lanes):
        if name not in placed:
            findings.append(f"{name} is passed to no body this script reaches")

    print(
        f"check_lane_members_read: {len(lanes)} lane struct(s), "
        f"{sum(len(lane.members) for lane in lanes.values())} member(s), "
        f"{len(contexts)} function template(s) and {len(generic)} lane seat(s) read "
        f"off {source_path.name} and {arithmetic_path.name}"
    )
    print(
        f"  {len(placed & set(lanes))} of {len(lanes)} struct(s) reach one of the "
        f"{len({seat for seat, _ in flows})} seat(s) a lane is read from"
    )
    print(
        f"  held as interface, because a detection idiom asks for the name and does "
        f"without it: {', '.join(sorted(held)) or 'none'}"
    )

    if findings:
        print(f"\ncheck_lane_members_read: {len(findings)} finding(s)\n", file=sys.stderr)

        for finding in findings:
            print(f"  {finding}", file=sys.stderr)

        return 1

    print("check_lane_members_read: every lane member is read by a body, or held as interface")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
