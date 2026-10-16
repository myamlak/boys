#!/usr/bin/env python3
r"""Every body that evaluates a Boys value must declare the bar it is held to and where that bar is checked.

WHY THIS EXISTS. One arithmetic - a seed and an upward recurrence in a named division form - is written
out many times in this tree, once per lane, per shape and per fit route. Nothing enforced that the copies
stay a family, so they drifted: one body was corrected while its siblings kept the arithmetic it was
corrected from, and no check noticed. Thirteen bodies were left carrying the uncorrected step; the one
that was corrected says so in its own comment ("the two things wrong with them repaired", "BoysAllNBodyRegionC's
step with the term region C drops").

The drifting numbers are NOT what went unseen. Every one of those bodies passes its published bound, and it
passes it comfortably, because the figure is stated against the ladder's first rung: at n = 32, x = 500 the
value is 4.45e-54 and the bound is 1e-13 * F_0(500) = 3.96e-15, which is 8.9e38 times it. A bound that large
cannot separate two arithmetics, so a per-path value test - the instrument that was prototyped first - reports
green on the whole defect class.

What went unseen is that thirteen bodies were never LISTED. That is an inventory failure, and this is the
inventory: every body that evaluates a Boys value is enumerated from the source, and each one must appear
in `tools/path-declarations.json` with the bar it is held to and the site that holds it there. A body the
source carries and the table does not is a finding - "this body exists and nothing declares what it
promises" - which is the sentence no value test can say.

WHAT IT CHECKS. Three sides, and a run that cannot read any of them refuses rather than passing.

  * THE BODIES, read from the tree. A body is an evaluation path when it carries the library's own ladder
    arithmetic, either of two ways, both read from the source rather than listed here:

      - it FORMS A STEP through a step primitive. The primitives are the definitions this tree names
        `*Step` and `*StepReciprocal` - seven of them, host and device, and the reader prints the set on
        every run so a definition it did not expect is in front of the reader who can judge it. This is a
        vocabulary and not a path list: a ladder body added tomorrow is found by calling one of these,
        without anything here being touched;
      - it FORMS A SEED from a half-sqrt-pi constant. The constants are the ones named `*HalfSqrtPi*`
        together with the ones the tree derives from those, transitively - which is how the native half
        lane's `kHalfNativeSeedConstant = F16FromDouble(kBoysHalfSqrtPi)` is read as a seed constant
        without this file knowing its name. This is the clause that sees a lane whose step is spelled with
        SIMD intrinsics, or with another file's primitives, and which the first clause cannot.

    What the two clauses read today: 1499 named bodies under `include/boys` and `src`, of which 23 are
    evaluation paths - 4 files carry them, and 19 of the 23 form a far seed, which is the one arithmetic
    written out nineteen times. The 24th body carrying the arithmetic is `UpwardStep`, a step primitive:
    the vocabulary is not a path, so it is printed as the vocabulary and not declared as one.

    The read audits itself before it reports. Every occurrence of the vocabulary in the tree must sit
    inside a body the reader named; a stray means the reader's map is not the tree's - a loop read as a
    body, say - and the check refuses rather than reporting a surface it did not read.

  * THE BAR. A row names the bar by its home in the tree - the file and the symbol that states it - and the
    check requires that symbol to be declared or defined in that file. The figure is NOT retyped here and is
    not compared: this check is about whether a body's promise is stated at all, and the figure's own checks
    are the ones that compare it.

  * THE VERIFICATION. A row names the site that holds the body to its bar - a file and a function - and the
    check requires that the site exists AND REACHES the body, by walking the calls this tree's own sources
    make. A site named but not reaching is the "bar nothing checks" this exists for, and it is a typed string
    that nothing would otherwise notice.

WHAT IT DOES NOT CATCH, stated rather than discovered later. This is an inventory, and an inventory is
silent about values:

  * TWO DECLARED BODIES THAT DISAGREE. If every body is listed, each with a bar that resolves and a site
    that reaches it, this check passes - however far apart their arithmetics have drifted. Two siblings may
    carry the same declared bar and compute different things; only a measurement separates them, and the
    measurement has to have a bar tight enough to bite (see the note above on why an absolute one cannot).
  * A BAR LOOSER THAN THE BODY NEEDS. A row whose bar resolves is a row whose bar resolves: this check does
    not ask whether the figure is the right one for the lane, only whether the tree states one.
  * A SITE THAT RUNS AND DOES NOT BITE. Reachability is by name through this tree's own sources. A site that
    calls the body but asserts nothing about it, or asserts against a cell where every lane agrees, reaches
    the body and passes here.
  * A BODY NO CLAUSE SEES. A lane that inlined its own recurrence and its own seed would carry neither a
    step call nor a seed constant and would be silent. Every lane in this tree forms its seed from a
    `HalfSqrtPi` constant, which is what makes the clause total today; the remedy for a lane that stopped is
    a step primitive it calls, not a spelling this check learns.
  * A GENERATED FILE. A source whose first line says the generator wrote it is not read: it is a table of
    numbers, it carries no body an entry is defined by, and reading it costs the run six megabytes of
    tables to find nothing in them. A body generated INTO a table would be outside this check.

Usage:
  python3 tools/check_paths_declared.py --check
  python3 tools/check_paths_declared.py --list                     # the enumeration, no table read
  python3 tools/check_paths_declared.py --control                  # plant the omission; require it caught
  python3 tools/check_paths_declared.py --root DIR --table FILE    # a table of your own to read

Exit status: 0 when every enumerated body is declared, every row names a body the source carries, every
bar resolves and every site reaches its body; 1 on any finding, and on a `--control` run that fails to
catch its own fixture; 1 when the tree cannot be read or the read finds no body at all - a check that can
pass by reading nothing is not a check.
"""

from __future__ import annotations

import argparse
import bisect
import json
import pathlib
import re
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
TABLE = pathlib.Path(__file__).resolve().parent / "path-declarations.json"
FORMAT = "boys.path-declarations/1"

# The sources the BODIES are read from: the library, and only the library. A test's body is not one of
# the bodies this check is about - it is the other side of the check, the side that holds a bar.
BODY_ROOTS = ("include/boys", "src")
BODY_SUFFIXES = (".hpp", ".cpp", ".cu")

# The sources the CALL GRAPH is read from. Wider than the bodies, because a verification site lives in the
# tests and in the benchmarks, and a site this graph cannot see would read as a site that does not reach.
GRAPH_ROOTS = ("include/boys", "src", "tests", "benchmarks")

# The definitions a step is formed through, by this tree's own naming: every step it writes is named for
# the step it is. Read off the source and printed by every run, so a definition the reader did not expect
# is visible on the run that read it. This is a vocabulary and not a list of paths: a ladder body added
# tomorrow is found by calling one of these, without anything here being touched.
STEP_NAME = re.compile(r"Step$|StepReciprocal$")

# The constant a far seed is formed from. Read by its own spelling and then TRANSITIVELY, so a constant
# derived from it - the native half lane's, which renames it - is a seed constant too.
SEED_CONSTANT_NAME = "HalfSqrtPi"

# A definition's head is not a definition's head when it opens one of these. `template` is absent on
# purpose: a function template's body is a body, and the clause is skipped instead.
NOT_A_BODY = re.compile(
    r"^\s*(if|for|while|switch|else|do|catch|try)\s*[\(\{]|"
    r"^\s*(namespace|struct|class|union|enum)\b"
)

# The words that can stand where a name stands and are not one. `for` and `while` are here as well as in
# NOT_A_BODY because a loop's head reaches this reader by a second way: a `while (f > a) {` whose
# semicolon-free header leaves the boundary before it far away, with a construct in between.
KEYWORDS = frozenset({
    "if", "for", "while", "switch", "else", "do", "catch", "try", "return", "sizeof", "alignof",
    "decltype", "static_assert", "assert", "requires", "operator", "new", "delete", "throw", "case",
    "namespace", "struct", "class", "union", "enum", "template", "using", "typedef", "public",
    "private", "protected", "co_return", "co_await", "co_yield", "constexpr", "consteval", "if",
})

# The test macros this tree writes a test with. A body written through one is named by the macro's own
# arguments rather than by the macro.
TEST_MACROS = frozenset({"TEST", "TEST_F", "TEST_P", "TYPED_TEST", "TYPED_TEST_P"})

IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def identifiers_in(text: str) -> set[str]:
    """Every identifier the text names, called or merely read.

    A seed is READ rather than called - `Half2::Broadcast(kHalfNativeSeedConstant)` names the constant
    and calls something else - so the clause that finds a seed cannot be a reading of the calls.
    """
    return set(IDENTIFIER.findall(text))


# A line comment, a block comment, a string literal and a character literal, as one alternation scanned
# left to right. One pass rather than four, because the four hide inside each other: a `//` inside a string
# is not a comment, and a `"` inside a comment is not a literal. The order of the alternatives IS the
# rule - whatever starts first at a position wins.
TOKEN = re.compile(
    r"//[^\n]*"                # a line comment
    r"|/\*.*?\*/"              # a block comment
    r'|"(?:[^"\\\n]|\\.)*"'    # a string literal
    r"|'(?:[^'\\\n]|\\.)*'",   # a character literal
    re.S,
)

KEEP_NEWLINE = re.compile(r"[^\n]")


def blank(text: str) -> str:
    """The text with comments and string and character literals replaced by spaces, newlines kept.

    Offsets survive, so a hit is still named by its line. This is the whole reason the reader is a reader
    of code rather than of text: without it every phrase in a docstring - and this library's headers are
    mostly docstring - reads as a call, a declaration or a constant, and a check that counted them would
    be reporting the prose.
    """
    return TOKEN.sub(lambda match: KEEP_NEWLINE.sub(" ", match.group(0)), text)


def line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


# The only characters the two maps below turn on. Both walk this instead of the text, which is what keeps
# a run over a two-hundred-kilobyte translation unit a walk over thousands of positions rather than
# hundreds of thousands: the characters between two of these move no depth and open no body.
SIGNIFICANT = re.compile(r"[(){};]")


def spans(text: str) -> list[tuple[int, int, int | None]]:
    """Every brace pair as (open, close, parent index), parents being the innermost containing pair."""
    found: list[list] = []
    stack: list[int] = []

    for match in SIGNIFICANT.finditer(text):
        c = match.group(0)

        if c == "{":
            found.append([match.start(), -1, stack[-1] if stack else None])  # type: ignore[arg-type]
            stack.append(len(found) - 1)
        elif c == "}" and stack:
            found[stack.pop()][1] = match.start()

    return [(o, c, p) for o, c, p in found]


def boundaries(text: str) -> list[int]:
    """Every `;`, `{` and `}` the text carries OUTSIDE parentheses, in order.

    The parenthesis is what makes this the tree's and not a text's: `for (int l = 1; l <= nmax; ++l)`
    carries two semicolons inside its own header, and a reader that took either for the end of a
    definition would name the loop's body `for` - which is how a check comes to declare a loop.
    """
    found: list[int] = []
    depth = 0

    for match in SIGNIFICANT.finditer(text):
        c = match.group(0)

        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif depth == 0:
            found.append(match.start())

    return found


class Source:
    """One file of the tree, blanked, with the two maps a body is read through."""

    def __init__(self, path: str, text: str) -> None:
        self.path = path
        self.text = text
        self.pairs = spans(text)
        self.marks = boundaries(text)

    def head_of(self, open_at: int) -> str:
        """The definition head ending at the body opening at `open_at`.

        The head runs from the last boundary before the body, which is what makes it one definition's
        rather than the file's. A comment is already blanked, so the doc block above a definition - which
        this tree writes at length - is not in it, and prose about a body cannot be read as its signature.
        """
        i = bisect.bisect_left(self.marks, open_at) - 1
        return self.text[self.marks[i] + 1 : open_at] if i >= 0 else self.text[:open_at]

    def name_of(self, open_at: int) -> str | None:
        """The name the body opening at `open_at` is defined under, or None where it defines none.

        The name is the identifier before the parenthesis whose close stands immediately before the body,
        with any trailing specifiers - `noexcept`, `const`, a trailing return type - walked over first. A
        control statement, a namespace, a class and a lambda define no name and answer None, which places
        whatever they contain in the body around them rather than in a body of their own.
        """
        text = self.text
        head = self.head_of(open_at)

        if NOT_A_BODY.match(head) or "[" in head.split("(")[0]:
            return None

        name = self._identifier(open_at)

        if name is None or name in KEYWORDS:
            return None

        # A test macro names a body too, and the name is its own arguments: `TEST(BoysTest, ScaledFar...)`
        # is one body called `BoysTest.ScaledFar...`. Reading it as `TEST` would merge every test in a file
        # into one node, and a site that reaches a body because SOME test in its file does is exactly the
        # false reach this check must not make.
        if name in TEST_MACROS:
            suite, _, case = head.split("(", 1)[-1].partition(",")

            if case:
                return f"{suite.strip()}.{case.split(',')[0].strip().rstrip(')').strip()}"

        return name

    def _identifier(self, open_at: int) -> str | None:
        """The identifier before the parenthesis closing immediately before `open_at`, if there is one.

        The walk starts at the body's `{` and steps back over whatever this tree writes between a
        definition's last `)` and its body - `noexcept`, `const`, a trailing return type - until it meets
        that `)`. It then steps back to the matching `(`, and the identifier before it is the name. A
        lambda has no identifier there and answers None, which leaves its body in the body around it.
        """
        text = self.text
        end = open_at - 1
        walked = 0

        while end >= 0 and walked < 12:
            while end >= 0 and text[end] in " \t\r\n":
                end -= 1

            if end >= 0 and text[end] == ")":
                break

            step = end

            while step >= 0 and (text[step].isalnum() or text[step] in "_>"):
                step -= 1

            if step == end:
                if end >= 0 and text[end] in "-*&:":
                    end -= 1
                    walked += 1
                    continue

                return None

            end = step
            walked += 1

        if end < 0 or text[end] != ")":
            return None

        depth = 0
        i = end

        while i >= 0:
            if text[i] == ")":
                depth += 1
            elif text[i] == "(":
                depth -= 1

                if depth == 0:
                    break

            i -= 1

        if i < 0:
            return None

        stop = i - 1

        while stop >= 0 and text[stop] in " \t\r\n":
            stop -= 1

        j = stop

        while j >= 0 and (text[j].isalnum() or text[j] == "_"):
            j -= 1

        if j == stop:
            return None

        return text[j + 1 : stop + 1]


class Body:
    """One function body in the tree: what it is called, where it stands, and what it carries."""

    def __init__(self, path: str, name: str, open_at: int, close: int, line: int, order: int) -> None:
        self.path = path
        self.name = name
        self.open_at = open_at
        self.close = close
        self.line = line
        self.order = order

    @property
    def key(self) -> str:
        """The body's identity: the file, the name, and - where a name is carried twice - its order.

        Keyed on the file and the name rather than on the line, because a line moves the moment anyone
        edits above it and a declaration that had to be re-typed for an unrelated edit is a declaration
        nobody keeps. A name carried twice in one file is two bodies here, distinguished by the order the
        reader meets them in, and reported as such.
        """
        return f"{self.path}::{self.name}#{self.order}" if self.order else f"{self.path}::{self.name}"


# One reading per (tree, roots) per run. A run asks for the bodies, then for the seed constants, then for
# the call graph, and the three together cover the same files: without this the tree is blanked three times
# over, and the coefficient header - 6 MB of generated tables, which this reader does walk - is the price.
_SOURCE_CACHE: dict[tuple, list[Source]] = {}

# The banner a generated file carries on its FIRST line, which is the reader's own marker rather than a
# guess: the generator writes it, and a header that opens with it is a table of numbers and not a body.
GENERATED = re.compile(r"//\s*Generated by\b")


def sources(root: pathlib.Path, roots: tuple[str, ...]) -> list[Source]:
    """Every first-party C++ source under the named roots, blanked and mapped."""
    key = (str(root), roots)

    if key in _SOURCE_CACHE:
        return _SOURCE_CACHE[key]

    found: list[Source] = []

    for relative_root in roots:
        base = root / relative_root

        if not base.is_dir():
            continue

        for path in sorted(p for p in base.rglob("*") if p.suffix in BODY_SUFFIXES):
            if "third_party" in path.as_posix():
                continue

            text = path.read_text(encoding="utf-8", errors="replace")

            if GENERATED.match(text):
                # A generated table is not a body: it says so on its first line, `gen_boys_coefficients.py`
                # writes that line, and nothing here can be a body an entry is defined by. Reading it anyway
                # costs the run six megabytes of tables for the privilege of finding nothing in them.
                continue

            found.append(Source(path.relative_to(root).as_posix(), blank(text)))

    _SOURCE_CACHE[key] = found
    return found


def read_bodies(root: pathlib.Path) -> list[tuple[Body, str]]:
    """Every named function body under the library's own sources, with the text it encloses."""
    found: list[tuple[Body, str]] = []
    seen: dict[str, int] = {}

    for source in sources(root, BODY_ROOTS):
        for open_at, close, _ in source.pairs:
            if close < 0:
                continue

            name = source.name_of(open_at)

            if name is None:
                continue

            identity = f"{source.path}::{name}"
            order = seen.get(identity, 0)
            seen[identity] = order + 1

            found.append((
                Body(source.path, name, open_at, close, line_of(source.text, open_at), order),
                source.text[open_at : close + 1],
            ))

    return found


def step_primitives(bodies: list[tuple[Body, str]]) -> set[str]:
    """The names a step is formed through: the definitions this tree writes `*Step` and `*StepReciprocal`.

    A vocabulary read off the tree rather than a path list: the bodies that form a ladder are found by the
    step they call, so a body added tomorrow is read without anything here being touched. The count is
    small enough to print on every run, which is what keeps a definition this check did not expect in front
    of the reader who can judge it.
    """
    return {body.name for body, _ in bodies if STEP_NAME.search(body.name)}


# A constant definition at any scope: `constexpr` is what this tree writes a constant with, and the
# initializer is kept so that a constant DERIVED from a seed is read as a seed. The initializer is capped
# so that a malformed read cannot run away over a whole file looking for a semicolon.
CONSTANT = re.compile(r"\bconstexpr\b[^;{}]*?\b(k[A-Za-z0-9_]*)\s*=\s*([^;]{0,400});", re.S)


def seed_constants(root: pathlib.Path) -> dict[str, str]:
    """The half-sqrt-pi constants, and every constant the tree derives from one, mapped to where it stands.

    Read by spelling first, then closed transitively over the tree's own constant definitions: the native
    half lane's seed constant is `F16FromDouble(kBoysHalfSqrtPi)` and carries the name of nothing, which is
    exactly the case a spelling-only read would lose.
    """
    definitions: dict[str, str] = {}
    homes: dict[str, str] = {}

    for source in sources(root, BODY_ROOTS):
        for match in CONSTANT.finditer(source.text):
            name = match.group(1)

            if name not in definitions:
                definitions[name] = match.group(2)
                homes[name] = f"{source.path}:{line_of(source.text, match.start(1))}"

    names = {name for name in definitions if SEED_CONSTANT_NAME in name}
    changed = True

    while changed:
        changed = False

        for name, initializer in definitions.items():
            if name in names:
                continue

            if any(part in names for part in IDENTIFIER.findall(initializer)):
                names.add(name)
                changed = True

    return {name: homes[name] for name in sorted(names)}


# An identifier, an optional template argument list and the parenthesis it is called with. The argument
# list is what makes this library's calls readable at all: `DivideStep<kForm>(a, x, invx)` is the form
# every step is written in, and `SingleOrderF32Body<FloatRouteFit<Policy::kRoute, ...>>(...)` nests one
# inside another, so a reader that stopped at the first `>` would miss the call and report a body no test
# reaches - the wrong direction for a check whose finding is "nothing verifies this".
# The identifier of a call, and only that: an identifier followed by the `(` it is called with or by the
# `<` a template argument list opens with. Anchoring on the two keeps the scan off every other word in the
# file, which over a whole tree is the difference between reading the calls and reading the prose.
CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?=[(<])")

# What may stand inside a template argument list: anything but a statement's own punctuation. A semicolon
# or a brace inside the angle group means the group is not a template argument list - `for (int i = 0; i < n;
# ++i)` is the case that matters, and it is the one that would otherwise swallow the loop's body. A call
# does write `kForm()` and `RegionBExp::kFast` inside one, so parentheses and scopes are allowed; the
# `(` that follows the group is what separates a template list from a comparison.
ARGUMENT_CHARS = re.compile(r"[^;{}]+")


def arc_over_template(text: str, start: int) -> int:
    """The offset just past a template argument list opening at `start`, or -1 where one does not.

    A `<` is a template bracket only where the call it is a call's follows it: that is the condition that
    separates `DivideStep<kForm>(a, x, invx)` from `a < b`, and it is settled by looking past the balanced
    angle group rather than by guessing at its contents. Stepping over a genuine `<` that opens no list is
    how a call reader loses the calls written after it, and a lost call is a body that reads as unverified.
    """
    close = matching_angle(text, start)

    if close < 0 or ARGUMENT_CHARS.fullmatch(text[start : close + 1]) is None:
        return -1

    after = close + 1

    while after < len(text) and text[after] in " \t\r\n":
        after += 1

    return after if after < len(text) and text[after] == "(" else -1


def calls_in(text: str) -> set[str]:
    """Every identifier the text calls, the template argument list between it and its parenthesis read."""
    found: set[str] = set()

    for match in CALL.finditer(text):
        i = match.end()

        if i < len(text) and text[i] == "<":
            stepped = arc_over_template(text, i)

            if stepped > 0:
                found.add(match.group(1))
                continue

        while i < len(text) and text[i] in " \t\r\n":
            i += 1

        if i < len(text) and text[i] == "(":
            found.add(match.group(1))

    return found


# How far a template argument list may run. Every list this tree writes is a few dozen characters; the
# bound is what stops the reader on a `<` that opens no list at all - `i < count` in a loop header - from
# scanning the rest of a translation unit looking for a `>` that closes nothing.
ANGLE_LIMIT = 400


def matching_angle(text: str, start: int) -> int:
    """The offset of the `>` closing the `<` at `start`, or -1 where none closes it near enough.

    The walk stops at a statement's own punctuation as well as at the limit. A `<` that opens no list is
    followed by an expression, and an expression does not run through a `;`, a brace or a bracket without
    the group having ended: reaching one is the answer that this is a comparison, and it is given without
    reading the rest of the file to be sure.
    """
    depth = 0
    stop = min(len(text), start + ANGLE_LIMIT)

    for i in range(start, stop):
        c = text[i]

        if c == "<":
            depth += 1
        elif c == ">":
            depth -= 1

            if depth == 0:
                return i
        elif depth == 1 and c in ";{}[]":
            return -1

    return -1


def evaluation_paths(root: pathlib.Path) -> tuple[list[Body], set[str], set[str], list[tuple[Body, str]]]:
    """Every body that carries the library's own ladder arithmetic, with the vocabulary that read them."""
    bodies = read_bodies(root)
    primitives = step_primitives(bodies)
    seeds = seed_constants(root)

    paths: list[Body] = []

    for body, text in bodies:
        # The primitives are the vocabulary and not a path: `DivideStep` is the axis every ladder reads a
        # form through, so a row declaring it would be declaring the arithmetic rather than a body that
        # delivers a value through it. They are printed as the vocabulary on every run instead.
        if body.name in primitives:
            continue

        called = calls_in(text)

        if called & primitives or identifiers_in(text) & set(seeds):
            paths.append(body)

    return paths, primitives, seeds, bodies


def graph(root: pathlib.Path) -> dict[str, set[str]]:
    """The calls this tree's own sources make, body by body, as name -> the names it calls.

    Read by name rather than by resolved symbol, which is the honest reading for a source-only walk over a
    template-heavy tree: every call this library makes to its own bodies is to a name written at the call
    site, and a name that two overloads share is a site reaching both - the looser direction, and the one a
    check about whether a site reaches a body must not fail in.
    """
    edges: dict[tuple[str, str], set[str]] = {}

    for source in sources(root, GRAPH_ROOTS):
        # The calls are read ONCE per file and then attributed by position, rather than once per body.
        # Bodies nest, so a per-body reading scans the file a depth's worth of times over, and the deepest
        # translation units in this tree are the ones with the most calls in them.
        marks: list[int] = []
        names: list[str] = []

        for match in CALL.finditer(source.text):
            i = match.end()

            if i < len(source.text) and source.text[i] == "<":
                i = arc_over_template(source.text, i)

                if i < 0:
                    continue

            while i < len(source.text) and source.text[i] in " \t\r\n":
                i += 1

            if i < len(source.text) and source.text[i] == "(":
                marks.append(match.start())
                names.append(match.group(1))

        for open_at, close, _ in source.pairs:
            if close < 0:
                continue

            name = source.name_of(open_at)

            if name is None:
                continue

            first = bisect.bisect_right(marks, open_at)
            last = bisect.bisect_left(marks, close)

            # Keyed on the file AND the name. Nearly every translation unit in this tree declares its own
            # `main`, and a graph keyed on the name alone merges them into one node that reaches the union
            # of what all of them reach - which is how a site reads as reaching a body it cannot call.
            edges.setdefault((source.path, name), set()).update(names[first:last])

    return edges


def reaches(edges: dict[tuple[str, str], set[str]], definitions: dict[str, set[str]],
            site: tuple[str, str], target: str) -> bool:
    """Whether the body at `site` reaches the definition named `target` through the tree's own calls.

    A call names a function and not a file, so a call to `BoysSingle` written in a test reaches every
    definition of that name - which is what the linker does, and the looser direction this must not fail
    in. The FILE is what keeps it from being too loose: `main` reached from `boys_boundary_standalone.cpp`
    is that file's `main` and not the accuracy gate's.
    """
    if site[1] == target:
        return True

    seen = {site}
    frontier = [site]

    while frontier:
        here = frontier.pop()

        for callee in edges.get(here, ()):
            if callee == target:
                return True

            for where in definitions.get(callee, ()):
                node = (where, callee)

                if node not in seen:
                    seen.add(node)
                    frontier.append(node)

    return False


def declared_symbol(root: pathlib.Path, where: str) -> str | None:
    """The reason a bar's home does not state the symbol, or None where it does.

    The home is a file and a name. The name must appear in it as something the file DEFINES or DECLARES -
    a definition, a constant, a declaration at any scope - and not merely as a word inside a sentence. A
    comment proves nothing, which is why the text is blanked before the search, and a bar whose only home
    is a sentence about it is the finding this clause exists for.
    """
    if "::" not in where:
        return "names no file and symbol (`<file> :: <symbol>`)"

    path_text, _, symbol = where.partition("::")
    path_text, symbol = path_text.strip(), symbol.strip()
    path = root / path_text

    if not path.is_file():
        return f"names {path_text}, which this tree does not carry"

    if not symbol:
        return "names no symbol"

    text = blank(path.read_text(encoding="utf-8", errors="replace"))

    if not re.search(r"\b" + re.escape(symbol) + r"\b", text):
        return (f"{path_text} carries no `{symbol}` outside its prose - a bar whose only home is a "
                f"sentence about it states no figure the tree can hand out")

    return None


def load_table(path: pathlib.Path) -> dict:
    """The declaration table, with a key written twice refused rather than silently kept last."""
    def pairs(pairs_list: list[tuple[str, object]]) -> dict:
        seen: dict = {}

        for key, value in pairs_list:
            if key in seen:
                raise ValueError(f"{key} is written twice, so one name is given two meanings")
            seen[key] = value

        return seen

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)


ROW_KEYS = {"id", "body", "bar", "verified_by"}
TOP_KEYS = {"format", "why", "paths"}


def read_rows(table: dict) -> tuple[list[dict], list[str]]:
    """The table's rows and the reasons it is malformed, each key read to its own meaning.

    An unknown key is refused rather than ignored. A field this check does not read is a claim about a body
    that nothing holds to the code - the row states something and no reader ever sees it - which is the same
    defect one level up from the one this file exists for.
    """
    problems: list[str] = []

    for key in table:
        if key not in TOP_KEYS:
            problems.append(f"the table carries the unknown key {key!r}")

    if table.get("format") != FORMAT:
        problems.append(f"the table is format {table.get('format')!r}, not {FORMAT!r}")

    if not isinstance(table.get("why"), str) or not table["why"].strip():
        problems.append("the table states no `why`: a table of promises with no account of what it is for "
                        "is a list a reader cannot tell was deliberate")

    rows = table.get("paths")

    if not isinstance(rows, list):
        problems.append("`paths` is not a list of rows")
        return [], problems

    kept: list[dict] = []

    for index, row in enumerate(rows):
        if isinstance(row, dict):
            kept.append(row)
        else:
            problems.append(f"row {index} is {type(row).__name__} rather than an object")

    return kept, problems


# What the stray-occurrence audit looks for: a call to a step primitive, or the name of a seed constant.
# The call form is the reader's own, so a primitive spelled with a template argument list is found; a seed
# is a name rather than a call, so it is searched for as a word.
OCCURRENCE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^;{}()]*>\s*)?\(")


def stray_occurrences(primitives: set[str], seeds: dict[str, str],
                      bodies: list[tuple[Body, str]]) -> list[str]:
    """Every place the ladder arithmetic is written outside a body this reader named.

    This is what keeps the enumeration honest. A reader that named a loop `for` would still find the step
    calls, still reach the enclosing body and still report a surface - one whose attribution is wrong in a
    way none of its own findings would show. A stray means the reader's map of the tree is not the tree's,
    so the check refuses rather than counting it.
    """
    named = [(body.open_at, body.close) for body, _ in bodies]
    stray: list[str] = []

    def placed(offset: int) -> bool:
        return any(open_at < offset < close for open_at, close in named)

    for body, text in bodies:
        for match in OCCURRENCE.finditer(text):
            if match.group(1) in primitives and not placed(body.open_at + match.start(1)):
                stray.append(f"{body.path}:{line_of(text, match.start(1))}")

        for name in seeds:
            for match in re.finditer(r"\b" + re.escape(name) + r"\b", text):
                if not placed(body.open_at + match.start()):
                    stray.append(f"{body.path}:{line_of(text, match.start())} (`{name}`)")

    return sorted(set(stray))


def check(root: pathlib.Path, table_path: pathlib.Path, verbose: bool
          ) -> tuple[int, list[str]]:
    """Enumerate the bodies, join them against the table, and report every finding."""
    paths, primitives, seeds, bodies = evaluation_paths(root)

    if not bodies:
        print(f"check_paths_declared: read no body from {', '.join(BODY_ROOTS)} under {root}. A check "
              f"that can pass by reading nothing is not a check", file=sys.stderr)
        return 1, []

    if not paths:
        print("check_paths_declared: read bodies and found no evaluation path among them - a defect of "
              "this check's own clauses and not a library with no ladder in it", file=sys.stderr)
        return 1, []

    # The read's own audit. Every occurrence of the vocabulary in the tree must sit inside a body the
    # reader named. An occurrence in no body is a parse that drifted - the reader naming a loop `for`, say -
    # and a check whose reading has drifted makes its findings about a surface it did not read. Refused
    # rather than warned about, because every finding here is a statement about what the source carries.
    stray = stray_occurrences(primitives, seeds, bodies)

    if stray:
        for where in stray[:12]:
            print(f"check_paths_declared: {where} carries the ladder arithmetic and this reader placed it "
                  f"in no body", file=sys.stderr)

        print(f"check_paths_declared: {len(stray)} occurrence(s) of the step or seed vocabulary sit "
              f"outside every body this reader named. The reading has drifted, so its enumeration is not "
              f"a statement about this tree", file=sys.stderr)
        return 1, [f"stray occurrence at {where}" for where in stray]

    if not table_path.is_file():
        print(f"check_paths_declared: {table_path} is missing, so no body below has anything declaring "
              f"what it promises", file=sys.stderr)
        return 1, [f"the table {table_path} is missing" for _ in paths[:1]]

    try:
        table = load_table(table_path)
    except (ValueError, json.JSONDecodeError, OSError) as error:
        print(f"check_paths_declared: {table_path} does not read: {error}", file=sys.stderr)
        return 1, [f"the table does not read: {error}"]

    rows, problems = read_rows(table)

    if problems:
        for problem in problems:
            print(f"check_paths_declared: {problem}", file=sys.stderr)

        return 1, problems

    edges = graph(root)
    definitions: dict[str, set[str]] = {}

    for (where, name) in edges:
        definitions.setdefault(name, set()).add(where)

    by_key = {body.key: body for body in paths}
    findings: list[str] = []
    declared: set[str] = set()

    for row in rows:
        unknown = sorted(set(row) - ROW_KEYS)

        if unknown:
            findings.append(f"the row {row.get('id', '(no id)')!r} carries {unknown}, which this check "
                            f"does not read: a field nothing holds to the code is a claim nothing tests")
            continue

        identity = row.get("body")

        if not isinstance(identity, str) or not identity:
            findings.append(f"the row {row.get('id', '(no id)')!r} names no body")
            continue

        if identity in declared:
            findings.append(f"{identity} is declared twice: one body under two bars, and only the last "
                            f"would be read")
            continue

        declared.add(identity)
        body = by_key.get(identity)

        if body is None:
            findings.append(f"{identity} is declared here and this tree's source carries no such body: a "
                            f"promise stated for an evaluation that is not there")
            continue

        # A body serves several regions and each of them states its own figure, so a bar is one home or a
        # list of them. A list is not a loosening: EVERY home in it must resolve.
        homes = row.get("bar")
        homes = [homes] if isinstance(homes, str) else homes

        if not isinstance(homes, list) or not homes or not all(
                isinstance(home, str) and home.strip() for home in homes):
            findings.append(f"{identity} declares no `bar`: a body with no stated bar is a body whose "
                            f"promise nothing states")
        else:
            for home in homes:
                reason = declared_symbol(root, home)

                if reason is not None:
                    findings.append(f"{identity} is held to the bar at `{home}`, and {reason}")

        sites = row.get("verified_by")
        sites = [sites] if isinstance(sites, str) else sites

        if not isinstance(sites, list) or not sites or not all(
                isinstance(site, str) and site.strip() for site in sites):
            findings.append(f"{identity} declares no `verified_by`: a bar nothing is named as checking is "
                            f"a bar nothing checks")
            continue

        for site in sites:
            where = site.partition("::")

            if not where[1] or not where[0].strip() or not where[2].strip():
                findings.append(f"{identity} is verified by `{site}`, which names no file and function "
                                f"(`<file> :: <function>`)")
                continue

            path_text, function = where[0].strip(), where[2].strip()

            # The site is looked up in the same reading that will follow its calls, so a name this check
            # can call and a name it cannot are the same judgement twice rather than two spellings of one.
            # A test macro's body is named by the macro's own arguments, which is what the table spells.
            if (path_text, function) not in edges:
                findings.append(f"{identity} is verified by `{site}`, and {path_text} defines no "
                                f"`{function}` outside its prose: a site that is not there checks nothing")
                continue

            if not reaches(edges, definitions, (path_text, function), body.name):
                findings.append(
                    f"{identity} is verified by `{site}`, and nothing {function} reaches is {body.name}: "
                    f"the site is named and the call that would have to hold the body to its bar is not "
                    f"there, so the bar is stated and unchecked"
                )

    # The enumeration's own direction: a body the source carries that no row declares. This is the finding
    # the whole check exists for, and it is stated in the terms an omission needs - the body exists and
    # nothing declares what it promises - rather than as a wrong number.
    undeclared = [body for body in paths if body.key not in declared]

    for body in sorted(undeclared, key=lambda b: (b.path, b.line)):
        callers = sorted({name for (_, name), callee in edges.items() if body.name in callee})
        reached = ", ".join(callers[:3]) + (f", +{len(callers) - 3} more" if len(callers) > 3 else "")
        findings.append(
            f"{body.path}:{body.line}: {body.name} evaluates a Boys value and nothing declares what it "
            f"promises - no bar and no site that holds it to one"
            + (f". Reached from {reached}" if callers else "")
            + f". Add a row to {table_path.name} naming this body, the bar it is held to, and the site "
              f"that checks it"
        )

    if verbose or not findings:
        print("check_paths_declared: the bodies, read from the source")
        print(f"  {len(bodies)} named body(ies) under {', '.join(BODY_ROOTS)}")
        print(f"  step primitives ({len(primitives)}): {', '.join(sorted(primitives))}")
        print(f"  seed constants ({len(seeds)}): {', '.join(sorted(seeds))}")
        print(f"  evaluation paths: {len(paths)}")
        print()
        print(f"check_paths_declared: the declaration table, {table_path.name}")
        print(f"  {len(rows)} row(s), {len(declared)} body(ies) named")

        if undeclared:
            print(f"  {len(undeclared)} body(ies) the source carries and nothing declares")

    if findings:
        print()
        print("check_paths_declared: a body that evaluates a Boys value must declare the bar it is held "
              "to and the site that holds it there. The figure this library publishes is stated "
              "against the ladder's first rung, and at large argument that reading is vacuous - at "
              "n = 32, x = 500 the value is 4.45e-54 against a 1e-13 * F_0(500) = 3.96e-15 bar. A "
              "value test therefore reports green on an arithmetic that has drifted; the inventory "
              "is what does not.", file=sys.stderr)
        print(file=sys.stderr)

        for finding in findings:
            print(f"  {finding}", file=sys.stderr)

        print(file=sys.stderr)
        print(f"check_paths_declared: {len(findings)} finding(s)", file=sys.stderr)
        return 1, findings

    print()
    print(f"check_paths_declared: all {len(paths)} evaluation path(s) declare a bar that resolves to a "
          f"symbol this tree states, and a site this tree's own calls reach")
    return 0, []


# The body the tree's own prose calls corrected: "the far region's third representation: the entry's own
# seed and ladder with the two things wrong with them repaired". The control declares THIS body and no
# other, which is the state the thirteen-site omission was - a correction that arrived with its own
# promise, and the copies it was corrected from left carrying none.
REPAIRED = "include/boys/boys_impl.hpp::CorrectedSeedLadder"

# A bar and a site for the control's one declared body. Both are the tree's and both resolve, so a failure
# of the control's run cannot be a row that is malformed: what it reports is the omission and nothing else.
#
# The site is the one this tree actually reaches that body from. The accuracy gate does NOT reach it - the
# far-representation arms are exercised by tests/boys_test.cpp alone, and a check that named the gate here
# would report the row rather than the omission. That is the check working, and the control must not paper
# over it by naming a site the tree does not have.
CONTROL_BAR = "tests/boys_accuracy_gate.cpp :: kBoundSingleC"
CONTROL_SITE = "tests/boys_test.cpp :: BoysTest.CorrectedSeedIsSelectableAndNotSelected"


def control(root: pathlib.Path) -> int:
    """Plant the omission and require it named: the corrected body declared, the siblings it came from not.

    The defect is reconstructed rather than described, and the reconstruction is the tree's own. The far
    ladder is one arithmetic written at many sites; one of its bodies carries the correction, and the
    siblings it was corrected from carry none. A table declaring that one body - with a bar that resolves
    and a site that reaches it, so nothing about the row itself can fail - is the state the omission was,
    and this control requires the check to name every sibling from it.

    It is not enough that the run fails. The control reads what was named and holds three things: the
    declared body is NOT among the findings, every sibling IS, and some finding carries the omission's own
    sentence. A check that failed for an unrelated reason would pass a weaker control, and would be
    evidence of nothing.
    """
    paths, _, seeds, bodies = evaluation_paths(root)
    text_of = {body.key: text for body, text in bodies}

    if not paths:
        print("check_paths_declared: the control read no evaluation path, so it has no defect to plant",
              file=sys.stderr)
        return 1

    if REPAIRED not in {body.key for body in paths}:
        print(f"check_paths_declared: the control names {REPAIRED} as the corrected body and this tree "
              f"carries no such body. The control's fixture is the tree's own prose, so either the body "
              f"moved or the prose describing it did, and this control will not guess which", file=sys.stderr)
        return 1

    # The siblings: the bodies that form a far seed, which is the arithmetic the corrected arm was corrected
    # from. Read by the same clause the check reads them by, so the set is not a second list to keep true.
    siblings = [body for body in paths
                if body.key != REPAIRED and identifiers_in(text_of[body.key]) & set(seeds)]

    if len(siblings) < 2:
        print(f"check_paths_declared: the control found {len(siblings)} far-ladder sibling(s) of "
              f"{REPAIRED}; the omission it plants is a set, and a set of one would pass for the wrong "
              f"reason", file=sys.stderr)
        return 1

    planted = {
        "format": FORMAT,
        "why": "the control's table: the corrected far-ladder body declared, the siblings it was "
               "corrected from declared by nothing.",
        "paths": [
            {
                "id": "control.the.one.corrected.body",
                "body": REPAIRED,
                "bar": CONTROL_BAR,
                "verified_by": CONTROL_SITE,
            }
        ],
    }

    print(f"check_paths_declared: control - the far ladder is one arithmetic at "
          f"{len(siblings) + 1} seed-forming bodies")
    print(f"check_paths_declared: control - the table declares {REPAIRED} and nothing else, with a bar "
          f"({CONTROL_BAR}) and a site ({CONTROL_SITE}) this tree carries")
    print(f"check_paths_declared: control - the {len(siblings)} sibling(s) must be named, and the declared "
          f"body must not be")
    print()

    with tempfile.TemporaryDirectory() as scratch:
        staged = pathlib.Path(scratch) / "path-declarations.json"
        staged.write_text(json.dumps(planted, indent=2), encoding="utf-8")
        code, findings = check(root, staged, verbose=False)

    named = "\n".join(findings)
    problems: list[str] = []

    if code == 0:
        problems.append("the run PASSED a table that declares one far-ladder body and leaves its "
                        "siblings undeclared")

    missed = [body.key for body in siblings if f"{body.path}:{body.line}: {body.name} " not in named]

    if missed:
        problems.append(f"{len(missed)} undeclared sibling(s) were not named: {', '.join(missed[:4])}")

    if REPAIRED in named:
        problems.append(f"the declared body {REPAIRED} was itself named as a finding, so the failure is "
                        f"about the row rather than about the omission")

    if not any("nothing declares what it promises" in finding for finding in findings):
        problems.append("no finding carried the omission's own sentence, so the failure came from "
                        "somewhere other than the omission")

    if problems:
        print("check_paths_declared: the control FAILED: " + "; ".join(problems), file=sys.stderr)
        print("  a check that cannot be made to fail on the defect it exists for is not evidence",
              file=sys.stderr)
        return 1

    print()
    print(f"check_paths_declared: control caught the omission - {REPAIRED} declared and passed, the "
          f"{len(siblings)} sibling(s) it was corrected from named, and the reason each was named is that "
          f"nothing declares what it promises")
    return 0


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero on a finding; the default run does the same, so a caller may "
                             "spell the invocation either way")
    parser.add_argument("--list", action="store_true",
                        help="print the enumeration and stop, reading no table")
    parser.add_argument("--control", action="store_true",
                        help="plant the omission - one far-ladder body declared, its siblings not - and "
                             "require the check to name the siblings")
    parser.add_argument("--root", type=pathlib.Path, default=REPO, help="the tree to read")
    parser.add_argument("--table", type=pathlib.Path, default=None, help="the declaration table")
    args = parser.parse_args()

    root = args.root.resolve()

    if not root.is_dir():
        print(f"check_paths_declared: {root} is not a directory", file=sys.stderr)
        return 1

    if args.control:
        return control(root)

    if args.list:
        paths, primitives, seeds, bodies = evaluation_paths(root)

        if not paths:
            print("check_paths_declared: read no evaluation path - a check that can pass by reading "
                  "nothing is not a check", file=sys.stderr)
            return 1

        print(f"{len(bodies)} named body(ies); step primitives ({len(primitives)}): "
              f"{', '.join(sorted(primitives))}")
        print(f"seed constants ({len(seeds)}): {', '.join(sorted(seeds))}")
        print(f"{len(paths)} evaluation path(s):")

        for body in sorted(paths, key=lambda b: (b.path, b.line)):
            print(f"  {body.path}:{body.line}: {body.name}")

        return 0

    table = args.table if args.table is not None else TABLE
    code, _ = check(root, table, verbose=False)
    return code


if __name__ == "__main__":
    sys.exit(main())
