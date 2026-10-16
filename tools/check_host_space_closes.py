#!/usr/bin/env python3
r"""Check that the host option space closes, from the accuracy gate's recorded run.

`tools/status.py` reports one arithmetic per option space, and three of them are not the same kind
of thing:

  * the HOST space closes from the accuracy gate's recorded run, which is committed and exists
    before any probe runs;
  * the DEVICE and PROBE spaces close from the option probes' own reports, and those reports are
    what a probe run for keeps PRODUCES.

That difference matters at exactly one moment: before a probe run. A gate that waits for the
device and probe spaces to close before letting a probe run is waiting for the thing it guards -
it can never open. This tool exists so the gate can name the half that is a precondition, and
leave the other half where it belongs: as the run's output, judged after the run.

It reads the recorded run and requires two things of it: the space's own arithmetic sums, and the
number certified equals the number of members. Both are read from the run's printed lines rather
than recomputed here, so a run that stops printing them fails rather than passing quietly.

Usage:

    python3 tools/check_host_space_closes.py
"""

import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
RUN = REPO / "tests" / "data" / "boys_accuracy_gate_run.txt"

COMBINATIONS = re.compile(r"COMBINATIONS:\s*(\d+)\s+of\s+(\d+)\s+member\(s\)")
ARITHMETIC = re.compile(r"the arithmetic:\s*([\d\s+]+)=\s*(\d+)")


def main() -> int:
    if not RUN.exists():
        print(f"check_host_space_closes: no recorded run at {RUN.relative_to(REPO)}")
        return 1

    text = RUN.read_text(encoding="utf-8", errors="replace")

    combinations = COMBINATIONS.search(text)
    if combinations is None:
        print("check_host_space_closes: the recorded run prints no COMBINATIONS line, so the "
              "space's size is unstated and this check cannot read it")
        return 1

    certified, total = (int(value) for value in combinations.groups())

    arithmetic = ARITHMETIC.search(text)
    if arithmetic is None:
        print("check_host_space_closes: the recorded run prints no arithmetic line, so the space's "
              "parts are unstated and this check cannot tell a closed space from an unread one")
        return 1

    terms = [int(term) for term in arithmetic.group(1).split("+")]
    stated_total = int(arithmetic.group(2))

    print("check_host_space_closes: the host option space, from the gate's recorded run")
    print(f"  certified   {certified} of {total}")
    print(f"  arithmetic  {' + '.join(str(t) for t in terms)} = {stated_total}")

    if sum(terms) != stated_total:
        print(f"  the parts sum to {sum(terms)} and the run states {stated_total}: the space is "
              f"NOT CLOSED - its parts do not add up to its own total")
        return 1

    if stated_total != total:
        print(f"  the arithmetic totals {stated_total} and the space has {total} member(s): the "
              f"space is NOT CLOSED - members are unaccounted for")
        return 1

    if certified != total:
        print(f"  {total - certified} member(s) of {total} are not certified and published: the "
              f"space is NOT CLOSED")
        return 1

    print("  every member is accounted for and certified: the host space is CLOSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
