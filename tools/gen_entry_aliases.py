#!/usr/bin/env python3
"""Derive which rows of the device option space are one arithmetic under two names.

The device option space offers 352 rows under the names of the `DeviceEntry` enumeration,
and 88 of them are a second name for an arithmetic another row already offers. What makes
them the same is not in the row: the rational route's two scheme names are two rows because
each states the name it was reached by, and the kernel they run is one; the uniform grid's
packing axis has one member, so its two rows launch one kernel. Every such row carries the
same precision, shape, question, lane, region-B exponential and bound as its twin - and so
does `all-orders-fp64-mono`, whose twin `all-orders-fp64` is *not* the same arithmetic. The
row cannot answer this. The entries' own code can: this tool follows each row's entry to the
kernel it launches, or to the device body it runs, and two rows that reach one kernel are one
arithmetic.

It writes that relation into `include/boys/boys_cuda_options.hpp` as
`DeviceEntryArithmeticOf`, between the markers it reads back under `--check`. The block is
generated and the prose around it is not: this tool refuses to touch a line outside them.

    python tools/gen_entry_aliases.py             # write the block
    python tools/gen_entry_aliases.py --check     # exit non-zero unless it is what the bodies say

`--options`, `--library`, `--kernel`, `--device` and `--host-library` read a side from
another file, which is how a moved forwarder is shown to fail as a negative control.

Exit status: 0 when the block matches the sources; 1 when it does not (and, without
`--check`, it has been rewritten to match); 2 when a row states no entry, or an enumerator
has no option row - a disagreement to look at rather than to derive through.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

OPTIONS = pathlib.Path("include/boys/boys_cuda_options.hpp")
LIBRARY = pathlib.Path("src/boys_cuda.cpp")
KERNEL = pathlib.Path("src/boys_cuda.cu")
DEVICE = pathlib.Path("include/boys/boys_cuda_device.hpp")
HOST_LIBRARY = pathlib.Path("src/boys.cpp")

BEGIN = "// BEGIN GENERATED: the rows that are one arithmetic (tools/gen_entry_aliases.py)"
END = "// END GENERATED: the rows that are one arithmetic (tools/gen_entry_aliases.py)"

# A row of the enumeration, its doc on the `///<` that follows it and on the further `///<`
# lines where it wraps. The first row carries an explicit `= 0`.
ENUM_ROW = re.compile(r"^\s*(k\w+)\s*(?:=\s*\d+)?,\s*(?:///<.*)?$")
ENUM_DOC = re.compile(r"^\s*(k\w+)\s*(?:=\s*\d+)?,\s*///<\s*(.*)$")
ENUM_DOC_MORE = re.compile(r"^\s*///<\s*(.*)$")

# The entry a row's own doc names: `BoysCuda::Name at ...` for a launched row, `BoysDeviceName`
# for one a caller reaches from inside its own kernel.
DOC_HOST = re.compile(r"BoysCuda::(\w+)")
DOC_DEVICE = re.compile(r"(BoysDevice\w+)")

# A body is read from the head of its own line, so a call inside another body is not mistaken
# for a definition.
HOST_HEAD = re.compile(r"^BoysStatus BoysCuda::(\w+)\(")
DEVICE_HEAD = re.compile(r"^(?:__device__\s+)?BoysDeviceStatus\s+(BoysDevice\w+)\s*\(")
LAUNCHER_HEAD = re.compile(r"^extern \"C\" int (BoysCudaLaunch\w+)\(")
# A definition of the CPU surface, for the other half of the question.
HOST_ENTRY_HEAD = re.compile(r"^(?:inline\s+)?(?:constexpr\s+)?[A-Za-z_][\w:<>,& *]*?\b(Boys[A-Z]\w*)\s*\(")

RETURN_CALL = re.compile(r"return\s+([A-Za-z_]\w*)\s*\(")
RETURN_DEVICE = re.compile(r"return\s+(BoysDevice\w+)\s*(?:<[^>]*>)?\s*\(")
RUN_LAUNCH = re.compile(r"RunLaunch\(\s*(BoysCudaLaunch\w+)")
KERNEL_CALL = re.compile(r"\b(\w+Kernel)\s*<\s*([^<>]*(?:<[^<>]*>[^<>]*)*)\s*>")

# An option table row's fields, in the order the table states them. The name of the row is left
# out of the facts: two rows that are one arithmetic carry two names, and that is the relation.
TABLE_ROW = re.compile(
    r"\{(DeviceEntry::k\w+),\s*\"([^\"]+)\"\s*,\s*(DeviceOptionGroup::k\w+),\s*"
    r"(DeviceOptionPrecision::k\w+),\s*(DeviceOptionShape::k\w+),\s*"
    r"(DeviceOptionQuestion::k\w+),\s*(DeviceOptionAxis::k\w+),\s*(RegionBExp::k\w+),\s*"
    r"(BoysDeviceLane::k\w+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^}]+)\}"
)


class Row:
    """One `DeviceEntry` row: the name it is spelled with, the entry its doc names, its facts."""

    def __init__(self, name: str, surface: str, kind: str, facts: tuple[str, ...], order: int) -> None:
        self.name = name
        self.surface = surface
        self.kind = kind  # "host" for a launched row, "dev" for a device-callable one
        self.facts = facts
        self.order = order
        self.arithmetic = ""
        self.depth = 0


def bodies(text: str, head: re.Pattern[str]) -> dict[str, str]:
    """Each definition's body, from its head line to the next head line of the same shape."""
    lines = text.split("\n")
    starts = [(index, line) for index, line in enumerate(lines) if head.match(line)]
    found: dict[str, str] = {}

    for position, (line_no, line) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        found[head.match(line).group(1)] = "\n".join(lines[line_no:end])  # type: ignore[union-attr]

    return found


def enumerators_of(text: str) -> list[tuple[str, str]]:
    """Every row of the `DeviceEntry` enumeration, as (name, doc), the doc's lines joined."""
    start = text.index("enum class DeviceEntry : int {")
    body = text[start : text.index("\n};", start)]
    rows: list[tuple[str, str]] = []
    pending: tuple[str, list[str]] | None = None

    def keep() -> None:
        if pending is not None and pending[0] != "kCount":
            rows.append((pending[0], " ".join(pending[1]).strip()))

    for line in body.split("\n"):
        opening = ENUM_DOC.match(line)
        more = ENUM_DOC_MORE.match(line)

        if opening is not None:
            keep()
            pending = (opening.group(1), [opening.group(2).strip()])
        elif more is not None and pending is not None:
            pending[1].append(more.group(1).strip())
        else:
            keep()
            pending = None

    keep()

    return rows


def table_of(text: str) -> dict[str, tuple[str, ...]]:
    """Each row's own facts, as the option table states them, keyed by the row's name.

    Two of the table's columns are left out. The slug is the row's own name, and a pair of
    names is what this tool is deriving. The axis is which axis of the space the row varies
    *for a reader*: the uniform grid's rows state different axes there and launch one kernel,
    and it is the kernel that decides. Every remaining column is a guard, so that two rows
    that reach one kernel over different lanes, bounds or precisions are reported rather than
    merged quietly.
    """
    start = text.index("constexpr DeviceOptionInfo kDeviceOptions[]")
    body = text[start : text.index("constexpr bool DeviceOptionsAreInEnumeratorOrder()", start)]
    found: dict[str, tuple[str, ...]] = {}

    for match in TABLE_ROW.finditer(body):
        columns = match.groups()

        found[match.group(1).replace("DeviceEntry::", "")] = columns[2:6] + columns[7:]

    return found


def kernel_of(launcher: str, launchers: dict[str, str]) -> str:
    """The kernel a launcher launches, as its call reads: its name and the arguments handed to it."""
    body = launchers.get(launcher)

    if body is None:
        return ""

    calls = set(KERNEL_CALL.findall(body))

    return f"{list(calls)[0][0]}<{list(calls)[0][1]}>" if len(calls) == 1 else ""


def resolve_host(surface: str, host: dict[str, str], launchers: dict[str, str]) -> tuple[str, int]:
    """Where a launched row's entry ends: the kernel it reaches, or the entry it stops at."""
    name = surface
    seen: set[str] = set()
    depth = 0

    while name in host and name not in seen:
        seen.add(name)
        body = host[name]
        calls = set(RETURN_CALL.findall(body))
        calls.discard(name)

        if "RunLaunch" not in body:
            if len(calls) == 1:
                name = calls.pop()
                depth += 1
                continue

            return (f"entry:{name}", depth)

        handed = set(RUN_LAUNCH.findall(body))

        if len(handed) == 1:
            launcher = handed.pop()
            kernel = kernel_of(launcher, launchers)

            return (f"kernel:{kernel}" if kernel else f"launcher:{launcher}", depth + 1)

        # A body that launches more than one kernel is a template the row's own axis member
        # selects between: which one it runs is the row's, and the row states it.
        return (f"entry:{name}", depth)

    return (f"entry:{name}", depth)


def resolve_device(surface: str, device: dict[str, str]) -> tuple[str, int]:
    """Where a device-callable row's entry ends: the body it runs rather than forwards to."""
    name = surface
    seen: set[str] = set()
    depth = 0

    while name in device and name not in seen:
        seen.add(name)
        calls = set(RETURN_DEVICE.findall(device[name]))
        calls.discard(name)

        if len(calls) == 1:
            name = calls.pop()
            depth += 1
            continue

        return (f"body:{name}", depth)

    return (f"body:{name}", depth)


def forwarding_host_entries(text: str) -> dict[str, str]:
    """Every definition of the CPU surface whose body is one call to another entry of the surface.

    This is the host half of the same question the device space asks, and the answer this
    library gives is none: a host entry that forwarded would be a second name for an
    arithmetic the API already offers under its own.
    """
    found: dict[str, str] = {}
    definitions = bodies(text, HOST_ENTRY_HEAD)

    for name, body in definitions.items():
        if not name.startswith("Boys"):
            continue

        calls = set(RETURN_CALL.findall(body))
        calls.discard(name)

        if len(calls) == 1 and len([line for line in body.split("\n") if line.strip()]) < 12:
            target = calls.pop()

            if target in definitions:
                found[name] = target

    return found


def derive(
    options: pathlib.Path,
    library: pathlib.Path,
    kernel: pathlib.Path,
    device: pathlib.Path,
    host_library: pathlib.Path,
) -> tuple[list[Row], dict[str, str]]:
    """The rows, each resolved to the arithmetic it runs, and the CPU surface's forwarders."""
    text = options.read_text(encoding="utf-8", errors="replace")
    library_text = library.read_text(encoding="utf-8", errors="replace")
    host = bodies(library_text, HOST_HEAD)
    host.pop("InitializeTables", None)
    launchers = bodies(kernel.read_text(encoding="utf-8", errors="replace"), LAUNCHER_HEAD)
    device_bodies = bodies(device.read_text(encoding="utf-8", errors="replace"), DEVICE_HEAD)
    table = table_of(library_text)
    rows: list[Row] = []

    for order, (name, doc) in enumerate(enumerators_of(text)):
        facts = table.get(name)

        if facts is None:
            raise SystemExit(f"gen_entry_aliases: {name} has no row in the option table")

        launched = DOC_HOST.search(doc)
        in_kernel = DOC_DEVICE.search(doc)

        if launched is not None:
            arithmetic, depth = resolve_host(launched.group(1), host, launchers)
            row = Row(name, launched.group(1), "host", facts, order)
        elif in_kernel is not None:
            arithmetic, depth = resolve_device(in_kernel.group(1), device_bodies)
            row = Row(name, in_kernel.group(1), "dev", facts, order)
        else:
            raise SystemExit(f"gen_entry_aliases: {name} names no entry in its own doc: {doc!r}")

        row.arithmetic = arithmetic
        row.depth = depth
        rows.append(row)

    return rows, forwarding_host_entries(host_library.read_text(encoding="utf-8", errors="replace"))


def classes_of(rows: list[Row]) -> list[list[Row]]:
    """The rows that are one arithmetic, each class in the enumerator order of its first row.

    Two rows are one arithmetic when they reach one kernel (a launched row) or one device body
    (a device-callable one) over the same lane, precision, shape, question, region-B exponential
    and bound. The class's own name for that arithmetic is the row that reaches it in the fewest
    steps, and where two reach it in as few, the first of them in the enumerator order.
    """
    grouped: dict[tuple[str, ...], list[Row]] = {}

    for row in rows:
        grouped.setdefault((row.kind, row.arithmetic, *row.facts), []).append(row)

    return list(grouped.values())


def statement_of(rows: list[Row]) -> dict[str, str]:
    """Each alias row's own name, mapped to the row it is the same arithmetic as."""
    aliases: dict[str, str] = {}

    for group in classes_of(rows):
        if len(group) < 2:
            continue

        own = min(group, key=lambda row: (row.depth, row.order))

        for row in group:
            if row is not own:
                aliases[row.name] = own.name

    return aliases


def block_of(aliases: dict[str, str], rows: list[Row]) -> str:
    """The generated C++ block, as the header carries it."""
    lines = [
        BEGIN,
        "// Written from the entries' own bodies: each row below reaches the kernel (or the device",
        "// body) of the row on its right, and the two are one arithmetic under two names.",
        "// Regenerate with `python tools/gen_entry_aliases.py`; `--check` verifies this block.",
        "",
        "/// The row that is the same arithmetic as \\p entry under another name, or \\p entry when",
        "/// its name is that arithmetic's only one.",
        "///",
        "/// Two rows are one arithmetic when their entries reach one kernel (a launched row) or one",
        "/// device body (a device-callable one) over one lane, precision, shape, question, region-B",
        "/// exponential and bound. The relation is read from the library's own bodies rather than",
        "/// listed here: `tools/gen_entry_aliases.py` follows each row's entry to what it runs, and",
        "/// this block is what it found. The row an alias is of is the one whose entry reaches that",
        "/// arithmetic in the fewest steps, and the first such row in the enumerator order - the",
        "/// order a report prints its rows in.",
        "///",
        "/// \\param entry the option",
        "///",
        "/// \\returns the row this one is a second name of, or \\p entry itself",
        "///",
        "/// \\ingroup boys",
        "constexpr DeviceEntry DeviceEntryArithmeticOf(DeviceEntry entry) noexcept {",
        "    switch (entry)",
        "    {",
    ]

    for row in rows:
        if row.name in aliases:
            lines.append(f"        case DeviceEntry::{row.name}:")
            lines.append(f"            return DeviceEntry::{aliases[row.name]};")

    lines += [
        "        default:",
        "            return entry;",
        "    }",
        "}",
        "",
        "/// Whether following an alias twice is following it once: the relation maps the space onto",
        "/// the rows that are an arithmetic of their own.",
        "constexpr bool DeviceEntryArithmeticsAreClosed() noexcept {",
        "    for (int i = 0; i < static_cast<int>(DeviceEntry::kCount); ++i)",
        "    {",
        "        const DeviceEntry arithmetic = DeviceEntryArithmeticOf(static_cast<DeviceEntry>(i));",
        "",
        "        if (DeviceEntryArithmeticOf(arithmetic) != arithmetic)",
        "        {",
        "            return false;",
        "        }",
        "    }",
        "",
        "    return true;",
        "}",
        "",
        "static_assert(DeviceEntryArithmeticsAreClosed(),",
        '              "an alias names a row that is itself an alias: the generated block in "',
        '              "boys_cuda_options.hpp is stale, run tools/gen_entry_aliases.py");',
        "",
        "/// How many of the space's rows are an arithmetic no other row offers.",
        "constexpr std::size_t DeviceEntryDistinctArithmeticCount() noexcept {",
        "    std::size_t distinct = 0;",
        "",
        "    for (int i = 0; i < static_cast<int>(DeviceEntry::kCount); ++i)",
        "    {",
        "        const DeviceEntry entry = static_cast<DeviceEntry>(i);",
        "",
        "        distinct += DeviceEntryArithmeticOf(entry) == entry ? std::size_t{1} : std::size_t{0};",
        "    }",
        "",
        "    return distinct;",
        "}",
        "",
        "/// How many of the space's rows are a second name for an arithmetic another row offers.",
        "constexpr std::size_t DeviceEntryAliasRowCount() noexcept {",
        "    return static_cast<std::size_t>(DeviceEntry::kCount) - DeviceEntryDistinctArithmeticCount();",
        "}",
        END,
    ]

    return "\n".join(lines)


def block_aliases(text: str) -> dict[str, str]:
    """The aliases the header's own block states, read back rather than assumed from the write."""
    if BEGIN not in text or END not in text:
        return {}

    block = text.split(BEGIN, 1)[1].split(END, 1)[0]
    stated: dict[str, str] = {}
    row: str | None = None

    for line in block.split("\n"):
        case = re.match(r"\s*case DeviceEntry::(\w+):\s*$", line)
        answer = re.match(r"\s*return DeviceEntry::(\w+);\s*$", line)

        if case is not None:
            row = case.group(1)
        elif answer is not None and row is not None:
            stated[row] = answer.group(1)
            row = None

    return stated


def spliced(text: str, block: str) -> str:
    """The file with the generated block replaced, refusing a file that carries no markers."""
    if BEGIN not in text or END not in text:
        raise SystemExit(
            "gen_entry_aliases: the options header carries no generated block - write its markers "
            f"and the prose above them first:\n  {BEGIN}\n  {END}"
        )

    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)

    return head + block + tail


def summary(rows: list[Row], aliases: dict[str, str], forwarding: dict[str, str]) -> str:
    """The derivation as the report states it."""
    distinct = len(rows) - len(aliases)
    launched = len([row for row in rows if row.kind == "host"])
    in_kernel = len(rows) - launched
    lines = [
        f"  (DeviceEntryArithmeticOf, include/boys/boys_cuda_options.hpp): {len(rows)} row(s) of the "
        f"space = {distinct}",
        f"  distinct arithmetic + {len(aliases)} row(s) that are a second name of one already "
        f"counted, over",
        f"  {launched} launched and {in_kernel} device-callable row(s); the 3 division form(s) of "
        f"BoysDivisionForms() cross",
        f"  them as {len(rows) * 3} member(s) over {distinct * 3} distinct (arithmetic, form) "
        f"pair(s)",
        f"  the CPU surface (src/boys.cpp): {len(forwarding)} entry/entries forward to another "
        f"entry of the surface",
    ]

    for row in rows:
        if row.name in aliases:
            lines.append(f"    {row.name} is {aliases[row.name]}")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the block instead of writing it")
    parser.add_argument("--options", type=pathlib.Path, default=OPTIONS)
    parser.add_argument("--library", type=pathlib.Path, default=LIBRARY)
    parser.add_argument("--kernel", type=pathlib.Path, default=KERNEL)
    parser.add_argument("--device", type=pathlib.Path, default=DEVICE)
    parser.add_argument("--host-library", type=pathlib.Path, default=HOST_LIBRARY)
    parser.add_argument("--quiet", action="store_true", help="print no derivation")
    arguments = parser.parse_args()

    rows, forwarding = derive(
        arguments.options,
        arguments.library,
        arguments.kernel,
        arguments.device,
        arguments.host_library,
    )
    aliases = statement_of(rows)
    block = block_of(aliases, rows)
    text = arguments.options.read_text(encoding="utf-8", errors="replace")

    if not arguments.quiet:
        print(summary(rows, aliases, forwarding))

    if arguments.check:
        stated = block_aliases(text)
        added = sorted(set(aliases) - set(stated))
        removed = sorted(set(stated) - set(aliases))
        changed = sorted(name for name in set(aliases) & set(stated) if aliases[name] != stated[name])

        if not added and not removed and not changed and spliced(text, block) == text:
            print("gen_entry_aliases: the header's block is what the entries' own bodies say")
            return 0

        print(
            "gen_entry_aliases: the header's block is not what the entries' own bodies say - "
            "regenerate with `python tools/gen_entry_aliases.py`",
            file=sys.stderr,
        )

        for name in added:
            print(f"  the bodies reach one arithmetic and the block does not say so: {name}", file=sys.stderr)

        for name in removed:
            print(
                f"  the block says {name} is {stated[name]} and the bodies no longer reach one "
                f"arithmetic",
                file=sys.stderr,
            )

        for name in changed:
            print(
                f"  the block says {name} is {stated[name]} and the bodies make it {aliases[name]}",
                file=sys.stderr,
            )

        return 1

    written = spliced(text, block)

    if written != text:
        arguments.options.write_text(written, encoding="utf-8", newline="\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
