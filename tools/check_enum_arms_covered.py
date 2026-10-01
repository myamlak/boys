#!/usr/bin/env python3
r"""Check that every switch over a tier or a scheme names every enumerator of it.

`AccuracyTier` and `OrdersScheme` are each dispatched by hand-written `case` lists, and the whole
guard around them rests on a claim nothing proves: that the arm list is complete. The `static_assert`
beside those dispatches proves the NAMING MAP is total - every enumerator is named in `TierRung` or
`SchemeIsNamed` - and the arms read their rung through that map. It says nothing about the arms
themselves, and an enumerator named in the map but given no arm still falls through.

What falls through is not a crash. Every one of these switches ends in a body that answers with the
reference precision or the certified summation - the slowest, most accurate option - so a tier or a
scheme with no arm is served at full accuracy under its own name, silently. That is a substitution:
the caller named a rung and got another one, and the probe books the cell as measured.

On gcc and clang `-Wswitch` catches the omission, and this tree builds `-Werror`, so the ubuntu legs
go red. It catches only the switches that have NO `default` arm, which is not all of them:
`AllOrdersF64AtRung` and `AllOrdersF32AtRung` keep a `default: break;` for the values cast in from
outside the enumeration, and that arm swallows a new enumerator exactly as quietly. On MSVC at /W4
nothing catches it anywhere - C4062 is not in /W4 - and MSVC is the toolchain the release is
verified on. This check is what covers the gap: it is the compiler's own rule, applied to every
switch uniformly, including the two the compiler cannot see.

Usage:
  python3 tools/check_enum_arms_covered.py --check
  python3 tools/check_enum_arms_covered.py --root <dir> --scan <file>   # negative control
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

# The enumeration, and the file that declares it, as paths under --root.
ENUMS = {
    "AccuracyTier": "include/boys/boys.hpp",
    "OrdersScheme": "src/boys_orders_simd.hpp",
}

# Every file a switch over one of those two can appear in: the library, the probe, the headers a
# consumer compiles, and the tests that dispatch on a tier the way a consumer does.
SCAN = (
    "src/boys.cpp",
    "src/boys_orders_simd.cpp",
    "src/boys_probe.cpp",
    "include/boys/boys.hpp",
    "include/boys/boys_impl.hpp",
    "tests/consumer_option_space.cpp",
)

# The sentinel each enumeration ends with: "one past the last member, and not a member". A switch is
# not required to name it - it is not a rung and not a scheme, and the two batch dispatches answer
# it through their `default` arm instead - so it is excluded from the set that must be covered and
# reported separately when it is absent, because a switch that names it is the stronger shape.
SENTINEL = "kCount"

CASE = re.compile(r"\bcase\s+(\w+)::(\w+)\s*:")
DEFAULT = re.compile(r"(?<![\w.])default\s*:")


def blank_literals(text: str) -> str:
    """Comments and string/char literals replaced by spaces, newlines kept.

    Offsets survive, so a hit can still be named by line. Without this a `// the default:` trail and
    the probe's `"is the default: ..."` sentences both read as `default` arms.
    """
    out = list(text)
    i, n = 0, len(text)

    while i < n:
        c = text[i]

        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                out[i] = " "
                i += 1
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            while i < n and not (text[i] == "*" and i + 1 < n and text[i + 1] == "/"):
                if text[i] != "\n":
                    out[i] = " "
                i += 1
            for j in range(i, min(i + 2, n)):
                out[j] = " "
            i += 2
        elif c in "\"'":
            quote = c
            out[i] = " "
            i += 1
            while i < n and text[i] != quote:
                if text[i] == "\\" and i + 1 < n:
                    out[i] = " "
                    i += 1
                if text[i] != "\n":
                    out[i] = " "
                i += 1
            if i < n:
                out[i] = " "
            i += 1
        else:
            i += 1

    return "".join(out)


def matching_brace(text: str, start: int) -> int:
    """The offset of the `}` closing the `{` at `start`, or -1."""
    depth = 0

    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i

    return -1


def enumerators_of(text: str, name: str) -> list[str] | None:
    """Every enumerator `enum class <name>` declares, in order."""
    head = re.search(r"enum\s+class\s+" + re.escape(name) + r"\b[^{;]*\{", text)

    if head is None:
        return None

    end = matching_brace(text, head.end() - 1)

    if end < 0:
        return None

    found = []

    for part in text[head.end() : end].split(","):
        match = re.match(r"\s*(k\w+)", part)

        if match is not None:
            found.append(match.group(1))

    return found


def switches_in(text: str) -> list[tuple[int, int, int]]:
    """(offset of `switch`, offset of the body `{`, offset of its `}`) for every switch statement."""
    found = []

    for match in re.finditer(r"\bswitch\s*\(", text):
        depth = 0
        i = match.end() - 1

        while i < len(text):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1

        j = i + 1

        while j < len(text) and text[j] in " \t\r\n":
            j += 1

        if j >= len(text) or text[j] != "{":
            continue

        end = matching_brace(text, j)

        if end > 0:
            found.append((match.start(), j, end))

    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit non-zero on a gap")
    parser.add_argument("--root", type=pathlib.Path, default=REPO, help="the tree to read")
    parser.add_argument("--scan", type=pathlib.Path, action="append", help="a file to scan")
    args = parser.parse_args()

    scanned = tuple(args.scan) if args.scan else tuple(pathlib.Path(p) for p in SCAN)
    # Under --root, or a negative control reads the enumeration from the copy and the switches from
    # the tree it was meant to replace: the two halves then disagree about the same file, which
    # reports a gap that is not there and misses the one that is.
    scanned = tuple(p if p.is_absolute() else args.root / p for p in scanned)
    sources = {name: args.root / path for name, path in ENUMS.items()}

    missing_source = [str(p) for p in list(sources.values()) + list(scanned) if not p.is_file()]

    if missing_source:
        print(f"check_enum_arms_covered: missing {', '.join(missing_source)}", file=sys.stderr)
        return 1

    wanted: dict[str, set[str]] = {}

    for name, path in sources.items():
        text = blank_literals(path.read_text(encoding="utf-8", errors="replace"))
        found = enumerators_of(text, name)

        if not found:
            print(
                f"check_enum_arms_covered: read no enumerator of {name} from {path}. A check that "
                f"can pass by reading nothing is not a check",
                file=sys.stderr,
            )
            return 1

        wanted[name] = {row for row in found if row != SENTINEL}

    print(f"  enumerations read: {len(wanted)}")
    for name in sorted(wanted):
        print(f"    {name}: {len(wanted[name])} member(s) to cover, sentinel {SENTINEL} excluded")

    checked = 0
    failures = 0
    unguarded = []

    for path in scanned:
        text = blank_literals(path.read_text(encoding="utf-8", errors="replace"))

        for _, body_start, body_end in switches_in(text):
            body = text[body_start : body_end + 1]
            labels: dict[str, set[str]] = {}

            for match in CASE.finditer(body):
                labels.setdefault(match.group(1), set()).add(match.group(2))

            for name, named in labels.items():
                if name not in wanted:
                    continue

                checked += 1
                line = text.count("\n", 0, body_start) + 1
                gap = sorted(wanted[name] - named)
                has_default = DEFAULT.search(body) is not None

                if has_default:
                    # A `default` arm is what stops -Wswitch, and this check is the only
                    # instrument left for the switch that carries one.
                    unguarded.append(f"{path}:{line} ({name})")

                if gap:
                    failures += 1
                    print(
                        f"\ncheck_enum_arms_covered: {path}:{line} dispatches on {name} and names "
                        f"no arm for {', '.join(gap)}:",
                        file=sys.stderr,
                    )
                    print(
                        f"  the switch falls through to the body below it, so {', '.join(gap)} "
                        f"{'is' if len(gap) == 1 else 'are'} answered as the entry that body "
                        f"serves - the caller named one rung and is given another, and the probe "
                        f"books the cell as measured",
                        file=sys.stderr,
                    )

    if checked == 0:
        print(
            f"check_enum_arms_covered: read {len(scanned)} file(s) and found no switch over any "
            f"tracked enumeration. A check that can pass by reading nothing is not a check",
            file=sys.stderr,
        )
        return 1

    if unguarded:
        print(f"\n  switches no compiler's -Wswitch can see, this check the only one that does:")
        for where in unguarded:
            print(f"    {where}")

    if failures:
        return 1

    print(f"\ncheck_enum_arms_covered: all {checked} switch(es) over the tracked enumerations name "
          f"every member they are required to")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
