#!/usr/bin/env python3
"""Check the packed orders lane's two lists against each other.

The across-orders lane is a template instantiated once per cell - a (scheme,
multiplier, route, partition) combination on the double entry, a (scheme,
multiplier, route, budget, partition) combination on the single-precision one -
and the cells are written out twice: `extern template` declarations in
include/boys/boys_impl.hpp, so that a call site reaches the definition the
library already holds, and the definitions themselves in
src/boys_orders_simd.cpp. The two lists are hand-written and macro-expanded, and
a cell that falls out of one of them fails silently: declared but not
instantiated is an undefined symbol at a consumer's link with nothing naming the
cell, and instantiated but not declared is a second copy of the body compiled
into every translation unit that calls it.

That is not hypothetical. The single-precision lane's narrow partition is
instantiated at eleven multipliers on each of two routes and two schemes and
declared at none of them - 56 cells - and nothing in the tree compared the two
lists, so nothing could report it.

This script compares them. It reads both files, expands the macros the two
lists are written with, extracts the specializations of BoysAllOrdersPacked and
BoysAllOrdersF32Packed from each, and reports every cell that is in one list and
not the other. Exit status is 0 when the lists agree and 1 when they do not, so
it can be run as a check beside the other tools/*.py --check steps.

What it compares is the two lists **to each other**, never to the option space
the library publishes, and the difference matters when a check is believed: a
class of cell that neither list names is invisible here, and this script reports
agreement over whatever the two files happen to hold. What catches that class is
the link - a cell the compiler is told to expect and no translation unit defines
is an unresolved external at a consumer's build, with the cell named. So this
check is what holds the two lists in step; the link is what holds them complete,
and neither stands in for the other.

The expansion is a small processor for the subset of the preprocessor these two
files use - object-like and function-like `#define`, `#undef`, and invocation -
and it is deliberately not a preprocessor: `#if`/`#else` are not evaluated, and
the file's own guard is handled by requiring the branch markers this file is
written with and comparing the two branches' lists against each other, so a
branch that stops agreeing with its sibling is a failure here rather than a
union that hides it. A construct the script cannot read is an error naming the
construct, never a quiet pass.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HEADER = REPO / "include" / "boys" / "boys_impl.hpp"
SOURCE = REPO / "src" / "boys_orders_simd.cpp"

ENTRIES = ("BoysAllOrdersPacked", "BoysAllOrdersF32Packed")

# The branch markers src/boys_orders_simd.cpp is written with. Both branches
# instantiate the same cells on this revision, and the script compares them so
# that a revision where they stop agreeing says so here.
GUARD_OPEN = "#if BOYS_SIMD_X86"
GUARD_ELSE = "#else // BOYS_SIMD_X86"
GUARD_CLOSE = "#endif // BOYS_SIMD_X86"


class CheckError(Exception):
    """A construct the script cannot read. Never a quiet pass."""


def strip_comments(text: str) -> str:
    """Remove C++ comments, keeping string and character literals intact.

    The extraction below reads declarations out of prose-adjacent text, and the
    prose in both files names the entries this script looks for, so a comment
    left in place would be read as a declaration.
    """
    out = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == '"' or c == "'":
            quote = c
            out.append(c)
            i += 1
            while i < n:
                out.append(text[i])
                if text[i] == "\\" and i + 1 < n:
                    out.append(text[i + 1])
                    i += 2
                    continue
                if text[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _split_params(name: str, rest: str) -> tuple[list[str], str]:
    """Split `(a, b) body` into the parameter names and the body."""
    if not rest.startswith("("):
        raise CheckError(f"the definition of {name} has no parameter list")
    depth = 0
    for k, ch in enumerate(rest):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                params = [p.strip() for p in rest[1:k].split(",") if p.strip()]
                return params, rest[k + 1 :]
    raise CheckError(f"the definition of {name} has an unclosed parameter list")


def expand(text: str, where: str) -> str:
    """Expand the macros `text` defines and invokes, in file order."""
    macros: dict[str, tuple[list[str], str]] = {}
    pieces: list[str] = []

    def substitute(name: str, args: list[str]) -> str:
        params, macro_body = macros[name]
        if len(params) != len(args):
            raise CheckError(
                f"{where}: {name} is defined with {len(params)} parameters and "
                f"invoked with {len(args)}"
            )
        bound = macro_body
        # Longest name first, so `kMultiplier` is not substituted inside
        # `kMultiplierX` - none is written that way at this revision, and the
        # order costs nothing.
        for param in sorted(params, key=len, reverse=True):
            bound = re.sub(rf"\b{re.escape(param)}\b", args[params.index(param)], bound)
        return bound

    def run(pattern_text: str, depth: int) -> str:
        if depth > 16:
            raise CheckError(f"{where}: macro expansion did not terminate")

        def replace(match: re.Match[str]) -> str:
            name = match.group("name")
            if name not in macros:
                return match.group(0)
            inside = match.group("args")
            args: list[str] = []
            depth_arg = 0
            current = ""
            for ch in inside:
                if ch == "," and depth_arg == 0:
                    args.append(current.strip())
                    current = ""
                    continue
                if ch == "(":
                    depth_arg += 1
                elif ch == ")":
                    depth_arg -= 1
                current += ch
            args.append(current.strip())
            args = [a for a in args if a != ""]
            return substitute(name, args)

        # A function-like macro's arguments: one level of nesting is all these
        # files use, and the character class excludes the closing parenthesis.
        pattern = re.compile(r"\b(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\((?P<args>[^()]*)\)")
        produced = pattern.sub(replace, pattern_text)
        if produced == pattern_text:
            return produced
        return run(produced, depth + 1)

    # Line continuations first, so a definition written across several lines is
    # one logical line here as it is to the compiler.
    logical = text.replace("\\\n", " ").split("\n")

    for line in logical:
        stripped = line.strip()
        if stripped.startswith("#define"):
            rest = stripped[len("#define") :].strip()
            # A function-like definition has its parameter list immediately after
            # the name; anything else is object-like, however many parentheses
            # the body holds.
            match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", rest)
            if match is None:
                raise CheckError(f"{where}: cannot read the definition of {stripped!r}")
            name = match.group(0)
            after = rest[len(name) :]
            if after.startswith("("):
                macros[name] = _split_params(name, after)
            else:
                # Recorded, not used: the two lists are built from function-like
                # macros, and a name that takes no arguments has no invocation
                # for the expansion below to replace.
                macros[name] = ([], after.strip())
            continue
        if stripped.startswith("#undef"):
            macros.pop(stripped[len("#undef") :].strip(), None)
            continue
        # Expanded here rather than after the whole file is read: a definition is
        # in scope only until its `#undef`, and both files undefine their cell
        # macros after invoking them.
        pieces.append(run(line, 0))

    return "\n".join(pieces)


DECL = re.compile(
    r"\b(?P<extern>extern\s+)?template\s+void\s+"
    r"(?P<entry>BoysAllOrdersPacked|BoysAllOrdersF32Packed)\s*<(?P<args>[^<>]*)>"
)


def cells(text: str, where: str) -> dict[str, dict[str, set[str]]]:
    """The specializations each entry is declared or defined at, by entry name."""
    found: dict[str, dict[str, set[str]]] = {
        entry: {"extern": set(), "definition": set()} for entry in ENTRIES
    }
    for match in DECL.finditer(text):
        kind = "extern" if match.group("extern") else "definition"
        args = re.sub(r"\s+", " ", match.group("args")).strip(" ,")
        args = ", ".join(part.strip() for part in args.split(","))
        found[match.group("entry")][kind].add(args)
    for entry in ENTRIES:
        if not found[entry]["extern"] and not found[entry]["definition"]:
            raise CheckError(f"{where}: no specialization of {entry} was found at all")
    return found


def branches(text: str, where: str) -> list[tuple[str, str]]:
    """Split the source at its own architecture guard, for the two-lists check.

    The markers name the guard in a trailing comment, so this runs on the text
    as it stands in the file and not on the comment-stripped copy.
    """
    if GUARD_OPEN not in text:
        raise CheckError(f"{where}: the branch marker {GUARD_OPEN!r} is gone")
    _, rest = text.split(GUARD_OPEN, 1)
    if GUARD_ELSE not in rest:
        raise CheckError(f"{where}: the branch marker {GUARD_ELSE!r} is gone")
    x86, rest = rest.split(GUARD_ELSE, 1)
    if GUARD_CLOSE not in rest:
        raise CheckError(f"{where}: the branch marker {GUARD_CLOSE!r} is gone")
    scalar, _ = rest.split(GUARD_CLOSE, 1)
    return [("BOYS_SIMD_X86=1", x86), ("BOYS_SIMD_X86=0", scalar)]


def report(header_cells, source_cells) -> int:
    failures = 0
    for entry in ENTRIES:
        declared = header_cells[entry]["extern"]
        defined = source_cells[entry]["definition"]
        print(f"{entry}: {len(declared)} declared, {len(defined)} instantiated")
        for label, missing, where in (
            ("declared but never instantiated", declared - defined, HEADER.name),
            ("instantiated but never declared", defined - declared, SOURCE.name),
        ):
            if not missing:
                continue
            failures += len(missing)
            print(f"  {len(missing)} {label} (a cell named in {where} and not in the other):")
            for cell in sorted(missing):
                print(f"    <{cell}>")
        if not (declared - defined) and not (defined - declared):
            print("  the two lists agree")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="the form the other tools take: report and exit non-zero on a difference",
    )
    parser.parse_args()

    try:
        header = cells(
            strip_comments(expand(HEADER.read_text(encoding="utf-8"), HEADER.name)), HEADER.name
        )
        expanded = expand(SOURCE.read_text(encoding="utf-8"), SOURCE.name)
        source = cells(strip_comments(expanded), SOURCE.name)

        # The two architecture branches instantiate the same cells on this
        # revision, so the sets compared above are the file's whichever branch a
        # build takes. A revision where they stop agreeing is one this check has
        # to be told about rather than one whose union it reports as agreement.
        for label, chunk in branches(expanded, SOURCE.name):
            written = cells(strip_comments(chunk), f"{SOURCE.name} ({label})")
            for entry in ENTRIES:
                a = written[entry]["definition"]
                b = source[entry]["definition"]
                if a and a != b:
                    once = sorted(a - b)
                    twice = sorted(b - a)
                    raise CheckError(
                        f"{SOURCE.name} ({label}): this branch does not instantiate the cells the "
                        f"file's other branches do - {len(once)} only here, {len(twice)} only "
                        f"elsewhere (first of each: "
                        f"{once[0] if once else 'none'!r}, {twice[0] if twice else 'none'!r})"
                    )
    except CheckError as error:
        print(f"check_orders_packed_cells: {error}", file=sys.stderr)
        return 1

    failures = report(header, source)
    if failures:
        print(
            f"\n{failures} cell(s) differ between the two lists. A cell that is declared and not "
            f"instantiated is an undefined symbol at a consumer's link; a cell that is "
            f"instantiated and not declared is a copy of the body in every translation unit that "
            f"calls it. Add the cell to the list it is missing from.",
            file=sys.stderr,
        )
        return 1
    print("\nthe packed orders lane's declared and instantiated cell sets are the same")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
