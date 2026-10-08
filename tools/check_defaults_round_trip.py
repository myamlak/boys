#!/usr/bin/env python3
"""Evaluate the build-defaults round trip: emit a seam, point a build at it, run that build's suite.

WHY. A call site that names no policy resolves through the build-defaults seam,
`include/boys/boys_build_defaults.hpp`, and the `BOYS_BUILD_DEFAULTS` CMake option replaces that
seam wholesale for one build. What a release rests on is the whole path: that a seam an option
probe's run implies is a seam a build compiles and a suite passes under. Reading the header does
not decide it and neither does reading the emitter; only running the path does, which is what this
tool does.

WHAT IT DOES, in four steps, each printed with its own exit code:

    1. emit       write a seam header from `tests/data/boys_option_probe_report.txt`, the
                  committed option-probe report: one row per class the committed seam keys,
                  carrying the combination that report's run ranked first in that class;
    2. configure  `cmake -DBOYS_BUILD_DEFAULTS=<the emitted seam>` into a scratch build directory,
                  with `CMAKE_SKIP_INSTALL_RULES=ON`: the emitted seam is a file in the build tree,
                  and a tree that exports its targets refuses a build-tree path in the exported
                  include interface, which is a rule about packaging and not about defaults;
    3. build      build that tree;
    4. suite      `ctest` in that tree.

Exit 0 only when the configure, the build and the suite all succeed. The emitted seam's path and
its sha256 are printed before the build starts, so a reader can point a build of their own at the
same file and read the same suite off it.

THE CONTROL. `--control` plants a defect in the emitted seam - the first row's route cell renamed
to an enumerator the library does not have - and requires the build to fail on it. That build is
one target, `boys-consumer-header-only`: a translation unit that includes `<boys/boys.hpp>` and
links nothing, which is enough because the row list is expanded in every unit that includes that
header, so a seam the compiler refuses cannot reach the suite. A control that passed would mean
the round trip cannot see a seam no compiler accepts, which is worse than no control at all.

Usage:

    python tools/check_defaults_round_trip.py
    python tools/check_defaults_round_trip.py --control
    python tools/check_defaults_round_trip.py --report <file> --seam <file> --work <dir>

The work directory holds the emitted seam and the build trees; it is created if absent, and a
build tree already configured inside it is refused unless `--reuse` is given, so a run never
builds into a tree that someone else configured. A full run is a configure and a build of the
whole tree - minutes. The control's build is one translation unit.

Exit status: 0 when the round trip succeeded, and under `--control` when the planted defect was
caught; 1 when a step failed, and under `--control` when a seam no compiler accepts built; 2 when
the tool could not run - a named file does not exist, the report ranks no class, or the build
directory holds a configured tree and `--reuse` was not given.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPORT = REPO / "tests" / "data" / "boys_option_probe_report.txt"
SEAM = REPO / "include" / "boys" / "boys_build_defaults.hpp"
ACCURACY = REPO / "include" / "boys" / "accuracy.hpp"
WORK = REPO / "build-defaults-round-trip"

# The name the library includes the seam under, and the two directories a run writes into.
SEAM_NAME = "boys_build_defaults.hpp"
EMITTED_DIR = "emitted"
BUILD_DIR = "build"
CONTROL_BUILD_DIR = "build-control"

# The one target the control builds, and the defect it plants: the first row's route cell renamed
# to an enumerator no axis of the library has. A unit that expands the row list cannot compile it.
CONTROL_TARGET = "boys-consumer-header-only"
DEFECT_FROM = "FitRoute::kChebyshev"
DEFECT_TO = "FitRoute::kNoSuchRoute"

# A class's key: one precision and one question shape, spelled as prose in the report and as
# enumerator names in the seam.
PRECISION_CELL = {"fp64": "kFp64", "fp32": "kFp32", "fp16": "kFp16", "bf16": "kBf16"}
PRECISION_WORD = {cell: word for word, cell in PRECISION_CELL.items()}
SHAPE_CELL = {
    "single": "kSingle",
    "all-orders": "kAllOrders",
    "fixed-n": "kFixedN",
    "all-n": "kAllN",
    "all-n-at-orders": "kAllNAtOrders",
}
SHAPE_WORD = {cell: word for word, cell in SHAPE_CELL.items()}

# The seven names a replacement seam states: the host lane's five and the device lane's own two.
DEVICE_NAMES = ("BOYS_BUILD_DEFAULT_DEVICE_DIVISION_FORM", "BOYS_BUILD_DEFAULT_DEVICE_REGION_B_EXP")

# A class line of the report, as the option probe writes it: the class, the entries its run
# measured, the entry that run ranked first and the figure that entry took.
CLASS_LINE = re.compile(
    r"^  (?P<precision>fp64|fp32|fp16|bf16) "
    r"(?P<shape>single|all-orders|fixed-n|all-n|all-n-at-orders) +"
    r"possible .*?fastest (?P<option>\S+) at (?P<nanoseconds>[0-9.]+) ns/argument",
    re.MULTILINE,
)

# The sentence in a class's block that states the winner's own combination, and the cells in it.
COMBINATION = "the fastest option's own combination:"
CELL = re.compile(
    r"\b(FitRoute|EvalScheme|BoysBudget|PackAxis|FitGranularity|DivisionForm|RegionBExp)::(k\w+)"
)

# How the run reached a class's first place, in the report's own words, and the name the seam's
# marker carries for the same fact.
REACHED = (
    ("a measured ordering of equals", "ordered"),
    ("which was unanimous", "refined"),
    ("which had a plurality", "vote"),
)

# A row of a seam: the three key cells and the first three policy cells, which share one line of
# the macro. The budget is read from here because it is a property of the lane and not of a row.
SEAM_ROW = re.compile(
    r"^\s*X\((kHost|kDevice), (k\w+), (k\w+), (FitRoute::\w+), (EvalScheme::\w+), "
    r"(BoysBudget::\w+),",
    re.MULTILINE,
)

NAMES = re.compile(r"^#define (BOYS_BUILD_DEFAULT_[A-Z_]+) (\S+)$", re.MULTILINE)
HOST_EXP = re.compile(r"\bkDefaultHostRegionBExp\s*=\s*(RegionBExp::\w+)")

SEAM_PROSE = """#pragma once

/// \\file
/// The build-defaults seam the committed option-probe report implies, written by
/// `tools/check_defaults_round_trip.py`. A build pointed at this file through the
/// `BOYS_BUILD_DEFAULTS` CMake option compiles these choices instead of the committed seam's.
///
/// Every measured row is a figure from `tests/data/boys_option_probe_report.txt`: the row carries
/// the combination that report's run ranked first in its class, and the marker beside the row
/// carries the figure that combination took. A row is a measurement on the machine that report
/// describes and not a choice of the library's, so a report from a build pointed at this file
/// describes that machine.
///
/// WHAT A REPLACEMENT CARRIES, and this file is one: the seven names below - the host's five and
/// the device lane's own two - a row list, and no `BOYS_BUILD_DEFAULTS_COMMITTED`.

"""

ROWS_PROSE = """/// The classes this file sets a default for: **one row per class**, in the table's own format.
/// A measured row is one the report's run placed first in its class, and the combination below is
/// that row's own, cell for cell. A row marked a choice is one the report ranked no cell of: it
/// states the seven names above at its own lane's budget, and every cell is a choice rather than a
/// measurement.
#define BOYS_BUILD_DEFAULT_ROWS(X)\\
"""


def die(message: str) -> None:
    """Refuse to run, naming what is missing."""
    print(f"check_defaults_round_trip: {message}", file=sys.stderr)
    raise SystemExit(2)


def read(path: Path, what: str) -> str:
    """A file the run needs, or a refusal naming it."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        die(f"cannot read the {what} {path}: {error}")
        raise  # unreachable: `die` raises


def precision_word(cell: str) -> str:
    """A lane's precision cell as the report spells it: one lane is one class key either way."""
    return PRECISION_WORD[cell.replace("Device", "")]


def seam_names(text: str) -> dict[str, str]:
    """The seven names a seam states, by macro."""
    return dict(NAMES.findall(text))


def seam_rows(text: str) -> list[tuple[str, str, str, str]]:
    """Each row's (device, precision, shape, budget) cells, in the order the seam lists them."""
    return [(match[1], match[2], match[3], match[6]) for match in SEAM_ROW.finditer(text)]


def parse_report(text: str) -> dict[tuple[str, str], dict]:
    """What a report's run ranked first in each class it measured, by (precision, shape)."""
    classes = list(CLASS_LINE.finditer(text))
    ranked: dict[tuple[str, str], dict] = {}

    for index, match in enumerate(classes):
        stop = classes[index + 1].start() if index + 1 < len(classes) else len(text)
        block = text[match.start():stop]
        start = block.find(COMBINATION)
        cells = [] if start < 0 else [f"{kind}::{name}" for kind, name in CELL.findall(block[start:])]
        ranked[(match["precision"], match["shape"])] = {
            "cells": cells[:7],
            "nanoseconds": float(match["nanoseconds"]),
            "reached": next((name for phrase, name in REACHED if phrase in block), ""),
            "sorted": match["option"].endswith("-sorted"),
        }

    return ranked


def row_text(device: str, precision: str, shape: str, cells: list[str]) -> str:
    """One row of the row list, in the shape the seam's own macro takes."""
    route, scheme, budget, pack, granularity, division, exp = cells

    return (
        f"    X({device}, {precision}, {shape}, {route}, {scheme}, {budget},\\\n"
        f"      {pack}, {granularity}, {division},\\\n"
        f"      {exp})\\\n"
    )


def emit(report_text: str, seam_text: str, accuracy_text: str) -> str:
    """The seam header the committed report implies, as the text of a replacement seam."""
    names = seam_names(seam_text)
    rows = seam_rows(seam_text)
    ranked = parse_report(report_text)

    host = ("BOYS_BUILD_DEFAULT_FIT_ROUTE", "BOYS_BUILD_DEFAULT_EVAL_SCHEME",
            "BOYS_BUILD_DEFAULT_PACK_AXIS", "BOYS_BUILD_DEFAULT_FIT_GRANULARITY",
            "BOYS_BUILD_DEFAULT_DIVISION_FORM")

    for macro in host + DEVICE_NAMES:
        if macro not in names:
            die(f"the seam {SEAM} states no {macro}, which every replacement carries")

    if not rows:
        die(f"the seam {SEAM} carries no row list, so a replacement of it has no classes to key")

    if not ranked:
        die(f"the report {REPORT} ranks no class of any precision and shape")

    # The budget a lane's entries run at, read off the seam's own rows: it is a property of the
    # lane rather than of a row, so any row of the lane states it.
    budgets: dict[str, str] = {}

    for _device, precision, _shape, budget in rows:
        budgets.setdefault(precision, budget)

    host_exp = HOST_EXP.search(accuracy_text)

    if host_exp is None:
        die(f"the header {ACCURACY} states no kDefaultHostRegionBExp for a host row to fall back to")

    body: list[str] = []
    device_rows = [row for row in rows if row[0] == "kDevice"]
    device_marker = (
        "    /* a choice, not a measurement: no device run stands, so each of\n"
        f"       these {len(device_rows)} rows states the fallback names above at its own lane's "
        "budget */\\\n"
    )
    device_stated = False

    for device, precision, shape, _budget in rows:
        budget = budgets[precision]

        if device == "kDevice":
            if not device_stated:
                device_stated = True
                body.append(device_marker)

            cells = [names[host[0]], names[host[1]], budget, names[host[2]], names[host[3]],
                     names[DEVICE_NAMES[0]], names[DEVICE_NAMES[1]]]
            body.append(row_text(device, precision, shape, cells))
            continue

        winner = ranked.get((precision_word(precision), SHAPE_WORD[shape]))

        if winner is not None and len(winner["cells"]) == 7:
            reached = f", reached by {winner['reached']}" if winner["reached"] else ""
            sorted_note = " through the sorted-arguments call" if winner["sorted"] else ""
            marker = (f"    /* measured: {winner['nanoseconds']:.2f} ns per argument{reached}"
                      f"{sorted_note} */\\\n")
            body.append(marker)
            body.append(row_text(device, precision, shape, winner["cells"]))
            continue

        marker = (
            "    /* a choice, not a measurement: this report ranked no cell of this\n"
            "       class, so the row states the names above at this lane's budget and the\n"
            "       library's own region-B exponential */\\\n"
        )
        cells = [names[host[0]], names[host[1]], budget, names[host[2]], names[host[3]],
                 names[host[4]], host_exp[1]]
        body.append(marker)
        body.append(row_text(device, precision, shape, cells))

    text = SEAM_PROSE

    for macro in host + DEVICE_NAMES:
        text += f"#define {macro} {names[macro]}\n\n"

    text += ROWS_PROSE
    text += "".join(body)
    text += "\n"

    return text


def plant_defect(text: str) -> str:
    """The emitted seam with one row's route cell renamed: the defect the control requires caught."""
    anchor = text.find("BOYS_BUILD_DEFAULT_ROWS")

    if anchor < 0:
        die("the emitted seam carries no row list, so the control has no row to plant its defect in")

    found = text.find(DEFECT_FROM, anchor)

    if found < 0:
        die(f"the emitted seam carries no {DEFECT_FROM} in its row list for the control to rename")

    return text[:found] + DEFECT_TO + text[found + len(DEFECT_FROM):]


def run(step: str, command: list[str]) -> int:
    """Run one command, printing it and the exit code it returned."""
    print(f"\n-- {step}: {' '.join(command)}", flush=True)
    started = time.monotonic()
    code = subprocess.call(command)
    print(f"-- {step}: exit {code} after {time.monotonic() - started:.1f} s", flush=True)

    return code


def configure_command(options: argparse.Namespace, build: Path, seam: Path) -> list[str]:
    """The configure line both the round trip and the control run.

    Install rules are off: the seam named here is a file in the build tree, and exporting the
    library's targets refuses a build-tree path in the exported include interface. That refusal is
    about how a build is packaged, and this round trip is about the defaults it compiles.
    """
    command = ["cmake", "-S", str(REPO), "-B", str(build)]

    if options.generator:
        command += ["-G", options.generator]

    command += [
        f"-DCMAKE_BUILD_TYPE={options.build_type}",
        "-DCMAKE_SKIP_INSTALL_RULES=ON",
        f"-DBOYS_BUILD_DEFAULTS={seam}",
    ]

    return command


def guard(build: Path, reuse: bool) -> None:
    """Refuse to configure into a tree that already holds a configured build."""
    if (build / "CMakeCache.txt").exists() and not reuse:
        die(f"the build directory {build} holds a configured tree; name another --work directory "
            "or pass --reuse to build into it")


def round_trip(options: argparse.Namespace, build: Path, seam: Path) -> int:
    """Configure, build and run the suite against an emitted seam."""
    if run("configure", configure_command(options, build, seam)) != 0:
        return 1

    if run("build", ["cmake", "--build", str(build), "--config", options.build_type,
                     "--parallel"]) != 0:
        return 1

    if run("suite", ["ctest", "--test-dir", str(build), "--build-config", options.build_type,
                     "--output-on-failure"]) != 0:
        return 1

    return 0


def control(options: argparse.Namespace, build: Path, emitted: str) -> int:
    """Require one translation unit to refuse a seam the control has broken."""
    defective = build.parent / "boys_build_defaults_control.hpp"
    defective.write_text(plant_defect(emitted), encoding="utf-8", newline="\n")
    print(f"-- control: the defect is planted in {defective}")

    if run("control configure", configure_command(options, build,
                                                  defective)) != 0:
        print("check_defaults_round_trip: the configure refused the defective seam, so the build "
              "never read it and the control establishes nothing", file=sys.stderr)
        return 1

    if run("control build", ["cmake", "--build", str(build), "--config", options.build_type,
                             "--target", CONTROL_TARGET]) == 0:
        print("check_defaults_round_trip: the build accepted a seam whose row list names an "
              "enumerator the library does not have", file=sys.stderr)
        return 1

    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", default=str(REPORT),
                        help="the option-probe report a seam is emitted from")
    parser.add_argument("--seam", default=str(SEAM),
                        help="the seam this tree commits: a replacement carries its class list "
                             "and the names of its seven")
    parser.add_argument("--work", default=str(WORK),
                        help="the scratch directory the emitted seam and the build trees go into")
    parser.add_argument("--build-type", default="Release", help="the configuration to build")
    parser.add_argument("--generator", default=None,
                        help="the CMake generator, defaulting to the one CMake picks")
    parser.add_argument("--reuse", action="store_true",
                        help="build into a scratch tree that already holds a configured build")
    parser.add_argument("--control", action="store_true",
                        help="plant a defect in the emitted seam and require the build to fail")
    options = parser.parse_args(argv)

    report_path = Path(options.report).resolve()
    seam_path = Path(options.seam).resolve()
    work = Path(options.work).resolve()
    build = work / (CONTROL_BUILD_DIR if options.control else BUILD_DIR)

    guard(build, options.reuse)

    report_text = read(report_path, "option-probe report")
    seam_text = read(seam_path, "committed seam")
    accuracy_text = read(ACCURACY, "library header")

    emitted_text = emit(report_text, seam_text, accuracy_text)

    # The emitted seam is written as `boys/boys_build_defaults.hpp` under a directory of its own:
    # the include path a build gets is the header's name, and a directory holding one file is the
    # unit a reader can point `-DBOYS_BUILD_DEFAULTS` at.
    emitted_dir = work / "emitted"
    (emitted_dir / "boys").mkdir(parents=True, exist_ok=True)
    emitted_path = emitted_dir / "boys" / SEAM_NAME
    emitted_path.write_text(emitted_text, encoding="utf-8", newline="\n")

    written = emitted_path.read_bytes()
    rows = seam_rows(written.decode("utf-8"))
    measured = written.decode("utf-8").count("/* measured:")

    print(f"-- emit: {emitted_path}")
    print(f"-- emit: {len(rows)} row(s), {measured} of them measured")
    print(f"-- emit: sha256 {hashlib.sha256(written).hexdigest()}")

    if options.control:
        return control(options, build, emitted_text)

    return round_trip(options, build, emitted_path)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
