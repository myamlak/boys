#!/usr/bin/env python
"""A construct opened inside a conditional must close inside it, in every branch.

WHY THIS EXISTS. Twice on 2026-10-03 the same defect broke something, in two languages, and the
author's own build could not see either:

  * `tests/boys_muladd_route_simd_test.cpp` opened `namespace {` inside `#if BOYS_SIMD_X86` and closed
    it after the `#endif`. On x86 that balances. On arm64 the branch is skipped, the namespace is never
    opened, and the closing brace is `error: expected declaration before '}' token`. FIVE CI legs
    failed - macos arm64, linux-arm64 gcc, windows-arm64 msvc, and both arm64 option-matrix cells -
    and no x86 leg failed.
  * A `/// \\cond` ... `/// \\endcond` span in `include/boys/boys.hpp` contained an `#if` whose `#endif`
    sat outside the span, which doxygen reports as `more #if's than #endif's in \\cond..\\endcond
    section` and treats as fatal. That failed the Docs step of `linux-x86 gcc Release` for hours while
    the run was read by its NAME instead of its log.

The class is one thing: **a paired construct whose two halves can be separated by a preprocessor
directive is balanced only under the configuration the author compiled, and the author's
configuration is the one configuration that cannot show it.**

WHAT IT CHECKS, per file, and it is deliberately narrow so that a finding is real:

  1. **Brace and bracket balance per conditional branch.** For each `#if`/`#elif`/`#else`/`#endif`
     group at file scope, the body of each branch must balance `{`/`}` on its own. A branch that
     opens a namespace, class or function body and does not close it is reported.
  2. **`#if`/`#endif` balance inside a `\\cond` span.** Every `#if`, `#ifdef` or `#ifndef` opened
     between a `\\cond` and its `\\endcond` must be closed before the `\\endcond`.

It does NOT check anything else, and it says so rather than implying a clean run means more than it
does. Braces inside string literals and comments are stripped before counting, because a probe that
prints `"{"` is not an unbalanced brace.

Usage:  python tools/check_conditional_spans.py [--check] [files...]
Exit 0 when every span closes, 1 when one does not. No files given: the tracked C++ and headers.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_GLOBS = ("include", "src", "tests", "benchmarks")

# A line that opens or closes a conditional. Anchored: a `#if` mentioned inside a comment or a string
# is not a directive, and this file's own prose contains several.
DIRECTIVE = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b")
COND_OPEN = re.compile(r"^\s*///?\s*\\cond\b")
COND_CLOSE = re.compile(r"^\s*///?\s*\\endcond\b")


def strip_noise(text: str) -> str:
    """Remove comments and string/char literals, keeping newlines so line counts survive."""
    out = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == '"' or c == "'":
            quote = c
            i += 1
            while i < n and text[i] != quote:
                if text[i] == "\\":
                    i += 1
                i += 1
            i += 1
            out.append(" ")
        elif c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                if text[i] == "\n":
                    out.append("\n")
                i += 1
            i += 2
            out.append(" ")
        else:
            out.append(c)
            i += 1
    return "".join(out)


def balance(text: str) -> int:
    return text.count("{") - text.count("}")


def branch_findings(path: str, raw: str) -> list[str]:
    """Every conditional branch that does not balance its braces on its own.

    Nesting is handled with a stack: a nested group's body counts toward the branch that encloses it,
    which is what makes the arm64 case reportable. An earlier version bailed out on the first nested
    `#if` and reported THAT as a finding - five false positives on a clean tree, which is worse than
    no checker at all.
    """
    clean = strip_noise(raw)
    clean_lines = clean.split("\n")

    findings = []
    # frame: [branch_start_line_index, braces_so_far_in_this_branch]
    stack: list[list[int]] = []
    # Braces outside every conditional, so a branch that inherits an open construct from the shared
    # region is not blamed for it.
    shared = 0

    for i, line in enumerate(clean_lines):
        m = DIRECTIVE.match(line)
        if m:
            kind = m.group(1)
            if kind in ("if", "ifdef", "ifndef"):
                stack.append([i + 1, 0])
            elif kind in ("elif", "else"):
                if stack:
                    # CHECK before resetting. An earlier version reset the count here without
                    # looking at it, so only the LAST branch of a group was ever examined - and in
                    # the arm64 case the imbalance is in the `#if` branch, which is exactly the one
                    # that was being thrown away. The negative control caught it; nothing else would
                    # have.
                    start, delta = stack[-1]
                    if delta != 0:
                        findings.append(
                            f"{path}:{start}: the conditional branch ending at line {i} does not "
                            f"balance its braces ({delta:+d}) - a construct it opens is closed "
                            f"outside the branch, so it balances only on the configuration that "
                            f"compiles this branch")
                    stack[-1][1] = 0
                    stack[-1][0] = i + 1
            elif kind == "endif":
                if stack:
                    start, delta = stack.pop()
                    if delta != 0:
                        findings.append(
                            f"{path}:{start}: the conditional branch running to line {i} does not "
                            f"balance its braces ({delta:+d}) - a construct it opens is closed "
                            f"outside the branch, so it balances only on the configuration that "
                            f"compiles this branch")
            continue

        delta = line.count("{") - line.count("}")
        if not delta:
            continue
        if stack:
            for frame in stack:
                frame[1] += delta
        else:
            shared += delta

    for start, delta in stack:
        if delta != 0:
            findings.append(f"{path}:{start}: a conditional branch (never closed at end of file) "
                            f"does not balance its braces ({delta:+d})")

    return findings


def cond_span_findings(path: str, raw: str) -> list[str]:
    """Every `\\cond` span whose `#if`s do not all close before its `\\endcond`."""
    findings = []
    lines = raw.split("\n")
    open_at = None
    for i, line in enumerate(lines):
        if COND_OPEN.match(line):
            open_at = i
            continue
        if COND_CLOSE.match(line):
            if open_at is not None:
                open_depth = 0
                for j in range(open_at, i):
                    m = DIRECTIVE.match(lines[j])
                    if not m:
                        continue
                    if m.group(1) in ("if", "ifdef", "ifndef"):
                        open_depth += 1
                    elif m.group(1) == "endif":
                        open_depth -= 1
                if open_depth != 0:
                    findings.append(
                        f"{path}:{open_at + 1}: the \\cond span ending at line {i + 1} contains "
                        f"{open_depth:+d} unclosed #if/#endif - doxygen reports this as \"more #if's "
                        f"than #endif's in \\cond..\\endcond section\" and treats it as fatal")
            open_at = None
    return findings


def tracked_files() -> list[str]:
    try:
        done = subprocess.run(["git", "-C", REPO, "ls-files"], capture_output=True, text=True,
                              timeout=60)
        names = done.stdout.split("\n") if done.returncode == 0 else []
    except Exception:
        names = []
    keep = (".hpp", ".cpp", ".cu", ".cuh")
    return [os.path.join(REPO, n) for n in names
            if n.endswith(keep) and n.split("/")[0] in DEFAULT_GLOBS]


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    paths = [os.path.join(REPO, a) if not os.path.isabs(a) else a for a in args] or tracked_files()

    findings = []
    for path in paths:
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                raw = handle.read()
        except OSError:
            continue
        findings += cond_span_findings(path, raw)
        findings += branch_findings(path, raw)

    if findings:
        print("check_conditional_spans: a construct whose halves a conditional can separate")
        for f in findings:
            print(f"  {f}")
        print()
        print("  A construct opened inside a conditional must close inside it. Otherwise it balances")
        print("  only on the configuration that compiled the branch, which is the one configuration")
        print("  that cannot show the imbalance.")
        return 1

    print(f"check_conditional_spans: {len(paths)} file(s), every conditional branch balances its own "
          f"braces and every \\cond span closes its own conditionals")
    return 0


if __name__ == "__main__":
    sys.exit(main())
