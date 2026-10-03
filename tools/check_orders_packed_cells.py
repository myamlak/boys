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
not the other. A cell is the whole declaration and not the template arguments
alone: the parameter count and the exception specification are part of the
symbol, so two lists that agree on <args> and disagree on either are reported
apart rather than as agreement. Exit status is 0 when the lists agree and 1 when
they do not, so it can be run as a check beside the other tools/*.py --check
steps.

A specialization the script reads is one the compiler must see for the check to
mean anything, so a specialization written inside a conditional is an error
naming the directive, never a line read as live. The one conditional it does
evaluate is src/boys_orders_simd.cpp's architecture guard: the file is split at
its markers, and each of the pieces that come out - the two branches, and the
text outside the guard that a build carries either way - is read with no
conditional left in it. That was the hole this check left open: a cell moved
inside a conditional - by a guard a build turns off, or by an `#if 0` left in
while the option it served was being removed - was still counted as instantiated
here while the compiler emitted no symbol for it, which is the same undefined
reference at a consumer's link with this check reporting agreement.

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
and it is deliberately not a preprocessor: no conditional is evaluated, the
file's own guard is handled by requiring the branch markers this file is
written with and comparing the two branches' lists against each other, so a
branch that stops agreeing with its sibling is a failure here rather than a
union that hides it, and a conditional the script was not taught is refused
rather than read as live. A construct the script cannot read is an error naming
the construct, never a quiet pass.
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

# One directive each: the three that open a conditional, the two that continue
# one, and the one that closes it. A line is a directive when it starts with one
# of these and nothing else is read from it; the nesting is what the cells below
# are read against.
COND_OPEN = re.compile(r"#\s*(?:if|ifdef|ifndef)\b")
COND_MID = re.compile(r"#\s*(?:elif|else)\b")
COND_CLOSE = re.compile(r"#\s*endif\b")


def conditional_contexts(text: str, where: str) -> list[str | None]:
    """The innermost conditional directive each line sits inside, or None.

    The expansion above does not evaluate conditionals, so a specialization
    written inside one would otherwise be read as if the compiler saw it - the
    one way this script can report agreement over a cell a build does not carry.
    A directive that closes nothing, or one that is never closed, is an error:
    the file is then not the shape this script reads, and guessing where its
    conditionals end is the quiet pass both callers below exist to avoid.
    """
    contexts: list[str | None] = []
    stack: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if COND_OPEN.match(stripped) or COND_MID.match(stripped):
            if COND_OPEN.match(stripped):
                stack.append(stripped)
            elif not stack:
                raise CheckError(f"{where}: {stripped!r} continues no conditional")
            else:
                stack[-1] = stripped
            contexts.append(None)
            continue
        if COND_CLOSE.match(stripped):
            if not stack:
                raise CheckError(f"{where}: {stripped!r} closes no conditional")
            stack.pop()
            contexts.append(None)
            continue
        contexts.append(stack[-1] if stack else None)
    if stack:
        raise CheckError(f"{where}: {stack[-1]!r} is never closed")
    return contexts


def signature(text: str, start: int, where: str) -> tuple[int, bool]:
    """The parameter count and the exception specification after `start`.

    The specialization's symbol is its template arguments and its function type
    together, so a list that agrees on the arguments and disagrees on either of
    these declares a symbol the other list does not define. `start` is the index
    just past the specialization's `>`; the parameter list is the next thing
    there, and a declaration without one is an error rather than a cell with no
    signature. Parentheses nest - a pointer-to-function parameter is legal here -
    so the list is scanned to its own closing parenthesis rather than matched
    with a pattern.
    """
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] != "(":
        raise CheckError(f"{where}: the specialization at offset {start} has no parameter list")
    depth = 0
    end = -1
    for k in range(start, len(text)):
        if text[k] == "(":
            depth += 1
        elif text[k] == ")":
            depth -= 1
            if depth == 0:
                end = k
                break
    if end < 0:
        raise CheckError(f"{where}: an unclosed parameter list at offset {start}")
    inside = text[start + 1 : end]
    arity = 0 if not inside.strip() else 1 + inside.count(",")
    return arity, text[end + 1 :].lstrip().startswith("noexcept")


def cells(
    text: str, where: str, required: bool = True
) -> dict[str, dict[str, set[str]]]:
    """The specializations each entry is declared or defined at, by entry name.

    A cell is the whole declaration: the arguments, the parameter count and the
    exception specification. A specialization under any conditional is refused,
    because the compiler sees it only when that conditional holds and a list of
    what the compiler sees is the only one worth comparing; the caller that has a
    guard to evaluate hands in the pieces the split produced, which carry no
    directive of their own. `required` is for the pieces: an entry absent from one
    piece is not an error there, and the caller that assembles them says which
    entries must be present at all (see main).
    """
    found: dict[str, dict[str, set[str]]] = {
        entry: {"extern": set(), "definition": set()} for entry in ENTRIES
    }
    contexts = conditional_contexts(text, where)
    for match in DECL.finditer(text):
        line = text.count("\n", 0, match.start())
        context = contexts[line]
        if context is not None:
            raise CheckError(
                f"{where}: a specialization of {match.group('entry')} sits under {context!r}, a "
                f"conditional this script does not evaluate - the build may not see it, and "
                f"reading it as live is how a cell the library does not hold passes this check. "
                f"Evaluate that conditional here, or write the specialization outside it."
            )
        kind = "extern" if match.group("extern") else "definition"
        args = re.sub(r"\s+", " ", match.group("args")).strip(" ,")
        args = ", ".join(part.strip() for part in args.split(","))
        arity, noex = signature(text, match.end(), where)
        cell = f"{args} | {arity} parameter(s){' noexcept' if noex else ''}"
        found[match.group("entry")][kind].add(cell)
    for entry in ENTRIES:
        if required and not found[entry]["extern"] and not found[entry]["definition"]:
            raise CheckError(f"{where}: no specialization of {entry} was found at all")
    return found


def branches(text: str, where: str) -> tuple[str, list[tuple[str, str]]]:
    """Split the source at its own architecture guard, for the two-lists check.

    The markers name the guard in a trailing comment, so this runs on the text
    as it stands in the file and not on the comment-stripped copy. What comes
    back is the text outside the guard - before it and after it, the one region
    a build carries either way - and the two branch texts. Splitting here is what
    lets every conditional check below run with nothing allowed: the guard's own
    directive is consumed by the split, so a conditional still enclosing a
    specialization inside a piece is one this script does not evaluate, whether
    it nests inside a branch or wraps the guard.
    """
    if GUARD_OPEN not in text:
        raise CheckError(f"{where}: the branch marker {GUARD_OPEN!r} is gone")
    before, rest = text.split(GUARD_OPEN, 1)
    if GUARD_ELSE not in rest:
        raise CheckError(f"{where}: the branch marker {GUARD_ELSE!r} is gone")
    x86, rest = rest.split(GUARD_ELSE, 1)
    if GUARD_CLOSE not in rest:
        raise CheckError(f"{where}: the branch marker {GUARD_CLOSE!r} is gone")
    scalar, after = rest.split(GUARD_CLOSE, 1)
    return before + after, [("BOYS_SIMD_X86=1", x86), ("BOYS_SIMD_X86=0", scalar)]


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
        # The source's own architecture guard is the one conditional this script
        # evaluates, and the split is what evaluates it: the guard's directive is
        # consumed here, and every conditional left inside a piece is refused by
        # cells() rather than read as live. The text outside the guard is read as
        # a piece of its own - a build carries it either way - and the two
        # branches as the two lists a build can take.
        outside, chunks = branches(expanded, SOURCE.name)
        pieces = [("outside the guard", outside)] + chunks
        carried: list[dict[str, dict[str, set[str]]]] = []
        source = {entry: {"extern": set(), "definition": set()} for entry in ENTRIES}
        for label, piece in pieces:
            written = cells(strip_comments(piece), f"{SOURCE.name} ({label})", required=False)
            carried.append(written)
            for entry in ENTRIES:
                for kind in ("extern", "definition"):
                    source[entry][kind] |= written[entry][kind]
        for entry in ENTRIES:
            if not source[entry]["extern"] and not source[entry]["definition"]:
                raise CheckError(f"{SOURCE.name}: no specialization of {entry} was found at all")

        # The two architecture branches instantiate the same cells on this
        # revision, so the sets compared above are the file's whichever branch a
        # build takes. A revision where they stop agreeing is one this check has
        # to be told about rather than one whose union it reports as agreement,
        # and a branch that instantiates nothing is not an agreement either: a
        # build taking it would find none of that entry's symbols, which is the
        # same undefined reference this check exists to catch. Both are compared
        # against what a build taking that branch would have - the branch and the
        # text outside the guard - so a cell written outside the guard, which
        # every build has, is in both sides of the comparison.
        outside_cells = carried[0]
        for (label, _), written in zip(chunks, carried[1:]):
            for entry in ENTRIES:
                a = written[entry]["definition"] | outside_cells[entry]["definition"]
                b = source[entry]["definition"]
                if a != b:
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
