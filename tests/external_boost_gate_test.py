#!/usr/bin/env python3
"""Run the gate built with BOYS_GATE_EXTERNAL_BOOST and hold its two Boost.Math rows to the claim.

Both rows must print an exceeded verdict, name the audit's cell and the cpp_bin_float_100
reference under Boost 1.92, and the run must exit non-zero.

    python tests/external_boost_gate_test.py <path-to-boys-accuracy-gate>
"""

from __future__ import annotations

import re
import subprocess
import sys

ROWS = (
    ("airy/ai", "35.9", "boost.math.airy_ai_100eps"),
    ("airy/bi", "78.1252", "boost.math.airy_bi_100eps"),
)
SOURCE = "Boost.Math documentation, {page} page, accuracy section"
BOOST_VERSION = "1.92"
EXCEEDED = re.compile(r"\[\s*exceeded\s*\]", re.IGNORECASE)


def row_block(book: str, page: str) -> str | None:
    """The printed book entry for one source page, or None when it is absent."""
    marker = SOURCE.format(page=page)
    at = book.find(marker)
    if at < 0 or book.find(marker, at + 1) >= 0:
        return None
    start = book.rfind("\n  [", 0, at)
    if start < 0:
        return None
    end = book.find("\n  [", at)
    return book[start : len(book) if end < 0 else end]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: external_boost_gate_test.py <boys-accuracy-gate>", file=sys.stderr)
        return 1

    run = subprocess.run(
        [argv[1], "--strict"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    book = run.stdout
    findings: list[str] = []

    if run.returncode == 0:
        findings.append(
            "the gate exited 0: an exceeded documented claim must leave it non-zero"
        )

    evidence_lines: list[str] = []
    for page, named_x, row_id in ROWS:
        block = row_block(book, page)
        if block is None:
            findings.append(f"{row_id}: no book entry for the {page} page was printed")
            continue
        if not EXCEEDED.search(block):
            findings.append(f"{row_id}: the book entry does not carry an exceeded verdict")
        if named_x not in block:
            findings.append(
                f"{row_id}: the evidence does not name the audit's cell x={named_x}"
            )
        if "cpp_bin_float_100" not in block:
            findings.append(f"{row_id}: the evidence does not name the cpp_bin_float_100 route")
        if BOOST_VERSION not in block:
            findings.append(
                f"{row_id}: the evidence does not name the pinned Boost {BOOST_VERSION}"
            )
        if "eps at x=" not in block:
            findings.append(f"{row_id}: the evidence does not state a worst cell in eps units")
        for line in block.splitlines():
            if "cpp_bin_float_100" in line and "eps" in line:
                evidence_lines.append(line.strip())

    for line in evidence_lines:
        print(line)

    if findings:
        print(f"external-boost gate: {len(findings)} finding(s)", file=sys.stderr)
        for finding in findings:
            print(f"  - {finding}", file=sys.stderr)
        if run.stderr:
            print(run.stderr.rstrip(), file=sys.stderr)
        return 1

    print("external-boost gate: both rows exceeded, named, and referenced as claimed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
