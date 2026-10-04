#!/usr/bin/env python3
r"""Check that the device probe can measure every entry the option table declares.

The device probe's accounted space is the library's own option table, read at run time
(`DeviceOptionSpaceClosure` takes `rows = BoysDeviceOptions().size()`, src/boys_cuda_probe.cpp),
and that table is one row per `DeviceEntry` in enumerator order (`DeviceOptionsAreInEnumeratorOrder`,
src/boys_cuda.cpp, asserted at compile time). So the rows the probe accounts for are the enumeration's
own members and not a list the probe carries - and the two cannot disagree without the library
failing to compile.

**What can still disagree is whether a row can be measured at all.** Producing a figure needs an
arm: the probe reaches an entry through a hand-written switch in its own CUDA translation unit
(`LaunchLaunched` for the launched family, `LaunchInKernel` for the device-callable one), and an
entry with no arm reaches that switch's `default` and produces no figure. A row that is accounted
for and unarmed is the defect this check exists to find: the space counts it, the report owes it a
place, and the entry the caller can name is one the probe never runs.

Three sets are read and reconciled, all of them from the tree:

  * the enumeration - `include/boys/boys_cuda_options.hpp`, the members of `DeviceEntry` below
    `kCount`;
  * the table - `src/boys_cuda.cpp`, the rows of `kDeviceOptions` with the entry, the group and the
    precision each row books;
  * the arms - `src/boys_cuda_probe_kernels.cu`, the `case ProbeEntry::...` labels of each dispatch
    switch, with the preprocessor guard each sits under.

The group a row carries is the library's own statement of how it is reached, and it is what decides
which switch must arm it: a `kLaunched` row is measured by launching the library's kernel, a
`kDeviceCallable` one by running the caller's kernel with the entry in it. An arm under `#if
BoysFp16` is the fp16 seam's, and it is read against the rows whose own `built` cell is the seam's
constant rather than against a second list here.

Both directions are defects: an entry no arm reaches is unmeasurable, and an arm naming an entry the
enumeration does not declare is a switch arm for an option that does not exist.

Usage:
  python3 tools/check_device_probe_measures_every_entry.py --check
  python3 tools/check_device_probe_measures_every_entry.py --kernel <file>   # negative control
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

OPTIONS = pathlib.Path("include/boys/boys_cuda_options.hpp")
LIBRARY = pathlib.Path("src/boys_cuda.cpp")
KERNEL = pathlib.Path("src/boys_cuda_probe_kernels.cu")

# One enumerator of `DeviceEntry`: the name, and the explicit value the first row carries.
ENUMERATOR = re.compile(r"^\s*(k\w+)\s*(?:=\s*\d+)?,", re.M)
ENUM_BODY = re.compile(r"^enum class DeviceEntry.*?^};", re.S | re.M)

# One row of `kDeviceOptions`: the entry it books, the name it prints, the group it is reached by and
# the precision it computes in. The fields may wrap, so the rows are matched over the flattened
# table rather than line by line.
ROW = re.compile(
    r"\{\s*DeviceEntry::(\w+)\s*,\s*\"([^\"]*)\"\s*,\s*(?:/\*[^*]*\*/\s*)?DeviceOptionGroup::(\w+)\s*,"
    r"\s*DeviceOptionPrecision::(\w+)\s*,"
)

# A dispatch arm. The guard is the condition of the innermost `#if`/`#ifdef` in force when the label
# is read, and it is kept as its own text: an arm the seam's guard covers and an arm under some
# other condition are two different facts, and only the first is the one the seam's rows answer for.
ARM = re.compile(r"case\s+ProbeEntry::(\w+)\s*:")
SEAM_GUARD = "BoysFp16"
GUARD_OPEN = re.compile(r"^\s*#\s*(if|ifdef|ifndef)\b(.*)")
GUARD_ELSE = re.compile(r"^\s*#\s*else\b")
GUARD_CLOSE = re.compile(r"^\s*#\s*endif\b")
SWITCH = re.compile(r"switch\s*\(\s*entry\s*\)")
FUNCTION = re.compile(r"^(?:template\s*<[^>]*>\s*)?(?:int|void|bool)\s+(\w+)\s*\(")


class Arm:
    """One `case ProbeEntry::...` label, with the function and the guard it sits under."""

    def __init__(self, entry: str, function: str, guard: str | None) -> None:
        self.entry = entry
        self.function = function
        self.guard = guard

    @property
    def guarded(self) -> bool:
        return self.guard is not None

    @property
    def seam_guarded(self) -> bool:
        """Whether the seam's own condition is the innermost one over this arm."""
        return self.guard is not None and SEAM_GUARD in self.guard


def enumerators_of(path: pathlib.Path) -> list[str]:
    """Every member of `DeviceEntry` below `kCount`, in the enumeration's own order."""
    body = ENUM_BODY.search(path.read_text(encoding="utf-8", errors="replace"))

    if body is None:
        return []

    names = [match.group(1) for match in ENUMERATOR.finditer(body.group(0))]

    return [name for name in names if name != "kCount"]


def rows_of(path: pathlib.Path) -> list[tuple[str, str, str, bool]]:
    """Every row of `kDeviceOptions`: (entry, name, group, behind the fp16 seam)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    start = text.find("kDeviceOptions[] = {")

    if start < 0:
        return []

    # The seam's own cell: a row the build cannot serve carries the constant and its reason in the
    # last two fields. It is read as a name and not as a value, so the check never evaluates a
    # configuration - it states which rows the arms behind the same guard have to cover.
    seam = "kFp16Served"
    found: list[tuple[str, str, str, bool]] = []

    for match in ROW.finditer(text, start):
        entry, name, group, precision = match.groups()
        tail = text[match.end() : match.end() + 200]
        found.append((entry, name, group, seam in tail))

    return found


def arms_of(path: pathlib.Path) -> list[Arm]:
    """Every dispatch arm with the function and the guard it sits under."""
    found: list[Arm] = []
    guards: list[str] = []
    function = ""

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        opened = GUARD_OPEN.match(line)

        if opened is not None:
            guards.append(line.strip())
            continue

        if GUARD_ELSE.match(line):
            continue

        if GUARD_CLOSE.match(line):
            if guards:
                guards.pop()
            continue

        named = FUNCTION.match(line)

        if named is not None:
            function = named.group(1)

        for match in ARM.finditer(line):
            found.append(Arm(match.group(1), function, guards[-1] if guards else None))

    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit non-zero on a mismatch")
    parser.add_argument("--options", type=pathlib.Path, default=OPTIONS, help="the enumeration")
    parser.add_argument("--library", type=pathlib.Path, default=LIBRARY, help="the option table")
    parser.add_argument("--kernel", type=pathlib.Path, default=KERNEL, help="the probe's dispatch")
    args = parser.parse_args()

    for path in (args.options, args.library, args.kernel):
        if not path.is_file():
            print(f"check_device_probe_measures_every_entry: missing {path}", file=sys.stderr)
            return 1

    declared = enumerators_of(args.options)
    rows = rows_of(args.library)
    arms = arms_of(args.kernel)

    if not declared or not rows or not arms:
        print(
            f"check_device_probe_measures_every_entry: read {len(declared)} enumerator(s), "
            f"{len(rows)} row(s) and {len(arms)} arm(s). A check that can pass by reading nothing "
            f"is not a check",
            file=sys.stderr,
        )
        return 1

    table = [entry for entry, _, _, _ in rows]
    launched = {entry for entry, _, group, _ in rows if group == "kLaunched"}
    callable_ = {entry for entry, _, group, _ in rows if group == "kDeviceCallable"}
    seamed = {entry for entry, _, _, seam in rows if seam}

    by_entry = {entry: (name, group) for entry, name, group, _ in rows}

    print(f"  enumeration members (excl. kCount): {len(declared)}")
    print(f"  rows of the library's option table: {len(rows)}")
    print(f"  dispatch arms the probe carries:    {len(arms)}")

    failures = 0

    # The table is the enumeration, one row each, in the enumeration's order: the compile-time
    # assertion's own reading, so a tree whose build nobody ran is still checked.
    missed = [entry for entry in declared if entry not in table]
    extra = [entry for entry in table if entry not in declared]
    out_of_order = [
        entry for index, entry in enumerate(table) if index < len(declared) and entry != declared[index]
    ]

    if missed or extra or out_of_order:
        failures += len(missed) + len(extra) + len(out_of_order)

        for entry in missed:
            print(
                f"\n  {entry} is an enumerator of DeviceEntry and no row of kDeviceOptions books it, "
                f"so nothing accounts for it",
                file=sys.stderr,
            )

        for entry in extra:
            print(
                f"\n  the option table books {entry}, which the enumeration does not declare",
                file=sys.stderr,
            )

        for entry in out_of_order:
            print(
                f"\n  {entry} is booked out of the enumeration's order, and a report prints its rows "
                f"in that order",
                file=sys.stderr,
            )

    # Arms, per switch. A launched row is reached by launching the library's kernel and a
    # device-callable one by running the caller's kernel with the entry in it, so the two families
    # have two switches and an arm in the wrong one is no figure either.
    launched_arms = {arm.entry for arm in arms if arm.function == "LaunchLaunched"}
    kernel_arms = {arm.entry for arm in arms if arm.function == "LaunchInKernel"}
    everywhere = {arm.entry for arm in arms}

    for label, wanted, carried in (
        ("launched", launched, launched_arms),
        ("device-callable", callable_, kernel_arms),
    ):
        unarmed = sorted(wanted - carried)
        foreign = sorted(carried - wanted)

        if unarmed:
            failures += len(unarmed)
            print(
                f"\ncheck_device_probe_measures_every_entry: {len(unarmed)} {label} entr(ies) the "
                f"option table declares and no arm reaches:",
                file=sys.stderr,
            )

            for entry in unarmed:
                name, _ = by_entry.get(entry, (entry, ""))
                print(
                    f"  {entry} ('{name}') is row of the option table and the probe's {label} "
                    f"switch carries no arm for it, so the run reaches that switch's default and "
                    f"the row is offered with no figure",
                    file=sys.stderr,
                )

        if foreign:
            failures += len(foreign)
            print(
                f"\ncheck_device_probe_measures_every_entry: {len(foreign)} arm(s) in the {label} "
                f"switch name an entry that is not one of its rows:",
                file=sys.stderr,
            )

            for entry in foreign:
                print(
                    f"  {entry} is armed in {label}'s switch and the option table books it as "
                    f"something else",
                    file=sys.stderr,
                )

    undeclared = sorted(everywhere - set(declared))

    if undeclared:
        failures += len(undeclared)
        print(
            f"\ncheck_device_probe_measures_every_entry: {len(undeclared)} arm(s) name an entry the "
            f"enumeration does not declare:",
            file=sys.stderr,
        )

        for entry in undeclared:
            print(
                f"  {entry} is armed in the probe and no enumerator of DeviceEntry answers to it",
                file=sys.stderr,
            )

    # The fp16 seam: the arms the seam's guard covers are the rows whose own built cell is the
    # seam's constant, so a build with the seam closed loses exactly the arms whose rows it also
    # refuses. An arm under any OTHER condition is neither set: whether it survives a configuration
    # is a fact about that condition, and a reader cannot place the row by the seam alone.
    guarded = {arm.entry for arm in arms if arm.seam_guarded}
    foreign_guard = {arm.entry for arm in arms if arm.guarded and not arm.seam_guarded}

    over_guarded = sorted(guarded - seamed)
    under_guarded = sorted(seamed - guarded)

    print(f"\n  rows behind the fp16 seam:          {len(seamed)}")
    print(f"  arms behind the seam's own guard:   {len(guarded)}")
    print(f"  arms behind another condition:      {len(foreign_guard)}")

    if over_guarded or under_guarded or foreign_guard:
        failures += len(over_guarded) + len(under_guarded) + len(foreign_guard)

        for entry in over_guarded:
            print(
                f"\n  {entry} is armed only behind the seam's guard and its own row is not one the "
                f"seam closes, so a build with the seam closed loses an arm whose row it serves",
                file=sys.stderr,
            )

        for entry in under_guarded:
            print(
                f"\n  {entry} is a row the seam closes and no arm of it sits behind the seam's own "
                f"guard, so a build with the seam closed arms a row it refuses",
                file=sys.stderr,
            )

        for entry in sorted(foreign_guard):
            print(
                f"\n  {entry} is armed under a condition that is not the seam's, so whether the arm "
                f"survives a build is a fact about that condition and not about the row",
                file=sys.stderr,
            )

    if failures:
        return 1

    print(
        f"\ncheck_device_probe_measures_every_entry: the {len(declared)} declared entr(ies) are the "
        f"{len(rows)} row(s) the probe accounts for, and every one of them is armed - "
        f"{len(launched)} launched in LaunchLaunched, {len(callable_)} device-callable in "
        f"LaunchInKernel"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
