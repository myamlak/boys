#!/usr/bin/env python3
r"""Every class's every possible combination has an accuracy bound, stated and honoured.

A combination is ready to ship when it is implemented, measured, probed and default-capturable.
``tools/combination_matrix.py`` reads the first and third of those off the option probe's report
and ``tools/check_class_combinations.py`` holds that report's per-class counts to the number of
combinations a class can be instantiated at. **Nothing checked the bound.** A combination the
library carries but states no figure for, and a combination whose stated figure the code does not
keep, both read as fine to every check in ``tools/``.

What a bound is, in this library's own words. ``BoysAccuracyGuaranteed`` (``include/boys/boys.hpp``)
answers an ``AccuracyFigure`` for a combination of axes on a **lane**: ``value``, ``available``,
``source`` (the sentence the figure was read under) and ``reason`` (why none is available, where
none is). The figure is the lane's contract row plus the two terms a lane may carry beside it - the
plain reciprocal's own rounding (``plainAdditive``) and the region-B exponential's contribution
(``additive``, a term of the member the row names and answered only where that member is named).

**It does not take a ``Shape``.** A class is a ``(device, precision, shape)`` triple and the
guarantee is a ``(precision, five axes)`` query, so what the accessor can be asked is per-lane
where the question is per-class. Each class below therefore carries the sentence saying so, and the
per-class half of the answer - the axis tuple a class's own default policy resolves to, and the
figure that tuple carries - is read from ``DefaultPolicy`` and ``DefaultGuarantee``, which name the
class. The shape reaches the space in exactly one way: a shape that evaluates one order has no
ladder to fill the orders packing axis with, so that axis has one member there and not two, and
that exclusion is proved by compiling it rather than asserted.

WHAT IS READ FROM THE LIBRARY, AND WHAT IS NOT:

  * the classes are the ``(device, precision, shape)`` rows the default-policy table carries, read
    from ``BOYS_DEFAULT_POLICY_BUILD_ROWS`` and ``BOYS_DEFAULT_POLICY_BUILD_DEVICE_ROWS``;
  * which shapes are one-order shapes is a reading of each ``Shape`` enumerator's own documentation
    - the sentence the enumerator is declared with, not a list in this file;
  * the axis members are the library's own tables - ``BoysFitRoutes``, ``BoysEvalSchemes``,
    ``BoysPackAxes``, ``BoysFitGranularities``, ``BoysDivisionForms``, ``BoysRegionBExps`` - read
    by a driver compiled from those tables, never spelled out here;
  * the figures are what ``BoysAccuracyGuaranteed`` and ``DefaultGuarantee`` answer.

HOW THE FIGURES ARE OBTAINED, AND WHY THAT IS NOT A PROBE. This check compiles one small driver
against the revision's own sources and runs it once. The driver evaluates no F_n, times nothing and
emits no seam: it calls the two accessors above, of which the header says "it is a table read and
not a measurement: nothing is evaluated and nothing is timed", and prints what they answer. It is
neither ``boys-option-probe`` nor ``boys-device-probe`` and it does not measure the option space;
it reads one. The compile is the slow part, and a compile is not a probe.

The driver is a consumer translation unit - it includes ``<boys/boys.hpp>`` and nothing else of the
library - and the library is linked beside it from ``add_library(boys ...)`` in the revision's
``CMakeLists.txt``, read rather than written here. It compiles under the definition the target
publishes to consumers, ``BoysFp16=1``, and under the shipped default choices: ``BOYS_BUILD_DEFAULTS``
is empty in a default configure, so the committed ``boys/boys_build_defaults.hpp`` is what a class's
default policy resolves to, and the default axes are the macros of that file. A configure that names
a ``BOYS_BUILD_DEFAULTS_DIR`` shadows that header with one of its own, which moves what
``DefaultPolicy`` resolves to and with it the figures this check reports. The configure also
publishes a ``BOYS_SCALAR_CONTRACTS`` definition; at this revision the consumer test reads it and no
figure path does, so it is named here as a seam this check does not carry rather than one that moved
a number. A build carrying another default list reads its own figures through ``DefaultGuarantee``;
this check reports the shipped choices' and says so, which is what it can do without a build
directory that another writer may hold.

WHAT FAILS, AND WHY EACH ONE MUST:

  * a combination with no bound at all - the library refuses it and states its reason, and a
    combination nothing bounds is one no caller can rely on;
  * a combination whose bound the code does not honour - the accuracy gate's recorded run
    (``tests/data/boys_accuracy_gate_run.txt``, read from the same revision) states, for each
    combination it covers, the figure its lane publishes and how many of the cells it swept left
    that figure. A row with cells outside its bound is this check's loudest failure: a figure the
    library states and the code it ships does not keep;
  * a run that checked no class - a class list or an axis table read as empty is what a broken read
    looks like from the outside, and it must never read as a clean bill. The accessor's guard is
    exercised at a value outside the packing enumeration on every lane, and a run in which no
    combination anywhere was answered with a figure is refused as well, because a query that
    answers nothing passes every "the refusal was correct" test there is;
  * a lane table whose rows do not answer for the enumerators at their positions - every figure
    this check reports for a class is read at a row of that table, so a table ordered differently
    from the enumeration would have it report one lane's figure against another lane's class.

The recorded run covers the five axes ``(lane, route, scheme, partition, packing axis)`` and reads
every cell of that cross at each division form. **It covers the region-B exponential nowhere** -
no row of it and no column - so those combinations are reported as unchecked against measurement
rather than passed, on the row.

usage:
  check_combination_bounds.py [--revision REV] [--tree DIR] [--no-compile] [--show N]

`--show N` prints the first N of each class's combinations beside the counts. A failing run prints
every combination that made it fail, whole, whatever `--show` says: a truncated list of what is
wrong is the shape of report this check exists to replace.

Exit status is 0 only when every class's every fundamentally-possible combination has a stated
bound, no combination the recorded run covers leaves its bound, and the run checked at least one
class. It is 1 otherwise, and 1 when the revision could not be read at all.
"""

from __future__ import annotations

import argparse
import io
import itertools
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent

# The MSVC environment the rest of this repository's tools build under.
VCVARS = r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"

# The definitions the library's own target publishes (CMakeLists.txt, target `boys`): the fp16
# seam is PUBLIC, so this driver compiles under it exactly as a consumer does.
DEFINES = ["/DBoysFp16=1"]

# The paths the driver is compiled against, and the two records it is held to. `src/` is taken
# whole, and `CMakeLists.txt` with it, because the library's source list is read from its target
# rather than written here.
ARCHIVE = ["CMakeLists.txt", "include", "src", "tests/data/boys_accuracy_gate_run.txt"]

# The unit separator. The accessor's `source` and `reason` are prose sentences carrying commas,
# colons and parentheses, so a delimiter no sentence carries keeps one record to a line.
SEP = "\x1f"

# An enumerator of an enum class with the documentation `///<` puts on its own line. Only an
# enumeration's first member carries an initialiser here, so the initialiser is optional.
ENUMERATOR = re.compile(r"^\s*(k[A-Za-z0-9]+)\s*(?:=[^,\n]*?)?\s*,?\s*(?:///<\s*(.*))?$")

# The body of an `enum class NAME : ... { ... };`.
ENUM_BLOCK = r"enum class {name}\s*:\s*[\w:]+\s*\{{(.*?)\n\}};"

# The member whose own documentation names one order. A shape that evaluates one order has one
# order to put in a vector lane, so the axis that fills a lane with a ladder's orders has nothing
# to fill it with. This is the same reading, with the same sentence, that
# tools/check_class_combinations.py makes; the control below proves it rather than quoting it.
ONE_ORDER = re.compile(r"\bone order\b", re.I)
ORDERS_AXIS = "kOrders"

# The host library's own source list, as its target declares it. Read from the revision rather
# than written here, so a source added to the target is linked by this check without an edit.
LIBRARY = re.compile(r"add_library\(\s*boys\s+(.*?)\)", re.S)

# A class of the default-policy table: `X(kPrecision, kShape)` inside one of the two row lists.
CLASS_ROW = re.compile(r"X\(\s*(k[A-Za-z0-9]+)\s*,\s*(k[A-Za-z0-9]+)\s*\)")
HOST_ROWS = re.compile(r"#define\s+BOYS_DEFAULT_POLICY_BUILD_ROWS\(X\)(.*)$", re.M)
DEVICE_ROWS = re.compile(r"#define\s+BOYS_DEFAULT_POLICY_BUILD_DEVICE_ROWS\(X\)(.*?)\n"
                         r"BOYS_DEFAULT_POLICY_BUILD_DEVICE_ROWS\(", re.S)

# One row of the recorded run's cross:
#   "  fp64, chebyshev, split-clenshaw, shipped, arguments   170082   0   5e-14   5.5e-14   <state>"
CROSS_ROW = re.compile(
    r"^\s{2}(?P<lane>[a-z0-9-]+),\s*(?P<route>[a-z0-9-]+),\s*(?P<scheme>[a-z0-9-]+),\s*"
    r"(?P<partition>[a-z0-9-]+),\s*(?P<axis>[a-z0-9-]+)\s+"
    r"(?P<cells>[\d.]+(?:e[-+]?\d+)?)\s+(?P<outside>[\d.]+(?:e[-+]?\d+)?)\s+"
    r"(?P<delivered>\S+)\s+(?P<bound>\S+)\s", re.M)
CROSS_OPEN = "the combinations: every combination this library's own tables offer"
CROSS_CLOSE = "the division form: every cell of the cross above"

# The accessor table: the figure the run recorded for one combination per lane, with the sentence
# the accessor read it under. Its columns are `combination bound delivered row bound call measured
# state`, and `delivered` states the prose `no figure` on the two half lanes rather than a number.
ACCESSOR_ROW = re.compile(
    r"^\s{2}(?P<lane>[a-z0-9-]+),\s*(?P<route>[a-z0-9-]+),\s*(?P<scheme>[a-z0-9-]+),\s*"
    r"(?P<partition>[a-z0-9-]+),\s*(?P<axis>[a-z0-9-]+)\s+"
    r"(?P<bound>\S+)\s+(?P<delivered>\S+(?:\s\S+)?)\s+"
    r"(?P<rowbound>\S+)\s+(?P<call>\S+)\s{2,}(?P<state>\S.*)$", re.M)
ACCESSOR_OPEN = "state and the source the accessor read"

# The host entries, per shape and precision, for the compile control. The spellings are the
# library's own, as tools/check_class_combinations.py already instantiates them.
SUFFIX = {"kFp64": "", "kFp32": "F32", "kFp16": "F16"}
SCALAR = {"kFp64": "double", "kFp32": "float", "kFp16": "boys::F16"}
VALUE = {"kFp64": "1.5", "kFp32": "1.5f", "kFp16": "boys::F16(1.5f)"}
BUDGET = {"kFp64": "kFloat", "kFp32": "kFloat", "kFp16": "kFp16"}
CALL = {
    "kSingle": "boys::Boys{shape}{suffix}<{policy}>(4, x);",
    "kAllOrders": "boys::Boys{shape}{suffix}<{policy}>(4, x, out);",
    "kFixedN": "boys::Boys{shape}{suffix}<{policy}>(4, xs, out, 2, 1);",
    "kAllN": "boys::Boys{shape}{suffix}<{policy}>(4, xs, out, 2);",
    "kAllNAtOrders": "boys::Boys{shape}{suffix}<{policy}>(ns, xs, out, 2);",
}
# The lane the control is compiled at: a lane that carries an entry at every shape, as
# tools/check_class_combinations.py controls on.
CONTROL_LANE = "kFp32"


def git(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *arguments], capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def materialize(revision: str, destination: pathlib.Path) -> None:
    """`revision`'s headers, sources and recorded run, extracted under `destination`.

    The library is read from the revision rather than from a working tree, because a working tree
    can have a writer in it: what this check reports is a claim about a revision, and a claim about
    a file someone is editing is a claim about nothing.
    """
    archived = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", revision,
                               *ARCHIVE], capture_output=True)

    if archived.returncode != 0:
        raise SystemExit("check_combination_bounds: git archive "
                         f"{revision} failed: {archived.stderr.decode(errors='replace').strip()}")

    with tarfile.open(fileobj=io.BytesIO(archived.stdout), mode="r:") as tar:
        tar.extractall(destination, filter="data")


def enumerators(text: str, name: str) -> list[tuple[str, str]]:
    """The (member, documentation) pairs `enum class name` declares, in declaration order."""
    block = re.search(ENUM_BLOCK.format(name=name), text, re.S)

    if block is None:
        raise SystemExit(f"check_combination_bounds: the header declares no enum class {name}, so "
                         "it cannot be read from this revision")

    found = []

    for line in block.group(1).splitlines():
        match = ENUMERATOR.match(line)

        if match is not None:
            found.append((match.group(1), match.group(2) or ""))

    if not found:
        raise SystemExit(f"check_combination_bounds: enum class {name} declares no enumerator")

    return found


def one_order_shapes(header: str) -> set[str]:
    """The `Shape` enumerators whose own documentation names one order."""
    return {member for member, doc in enumerators(header, "Shape") if ONE_ORDER.search(doc)}


def library_sources(tree: pathlib.Path) -> list[str]:
    """The host library's own source list, as `add_library(boys ...)` declares it.

    The library is what its target says it is, so the driver is linked against that list rather
    than against the one translation unit whose accessors it calls: `src/boys.cpp` names backends
    that `src/boys_simd.cpp` defines, and a check that linked the first alone would not link.
    """
    cmake = tree / "CMakeLists.txt"

    if not cmake.exists():
        raise SystemExit(f"check_combination_bounds: no CMakeLists.txt at {cmake}, so the "
                         "library's own source list cannot be read")

    match = LIBRARY.search(cmake.read_text(encoding="utf-8", errors="replace"))

    if match is None:
        raise SystemExit("check_combination_bounds: CMakeLists.txt declares no `add_library(boys "
                         "...)`, so the library's own source list cannot be read")

    sources = match.group(1).split()

    if not sources:
        raise SystemExit("check_combination_bounds: `add_library(boys ...)` names no source")

    return sources


def class_rows(header: str) -> list[tuple[str, str, str]]:
    """The (device, precision, shape) triples the default-policy table carries.

    These are the classes this build has: a triple the table leaves clear is one whose
    `DefaultPolicy` asserts on, which is a build error rather than a class.
    """
    host = HOST_ROWS.search(header)
    device = DEVICE_ROWS.search(header)

    if host is None or device is None:
        raise SystemExit("check_combination_bounds: the header states no default-policy row list, "
                         "so the classes cannot be read from it")

    rows = [("kHost", precision, shape)
            for precision, shape in CLASS_ROW.findall(host.group(1))]
    rows += [("kDevice", precision, shape)
             for precision, shape in CLASS_ROW.findall(device.group(1))]

    if not rows:
        raise SystemExit("check_combination_bounds: the default-policy row list names no class")

    return rows


def batch(tree: pathlib.Path, body: str, timeout: float) -> subprocess.CompletedProcess:
    """One `cmd.exe` batch under this repository's MSVC environment, run in `tree`."""
    script = tree / "run.bat"
    script.write_text("@echo off\n"
                      f'call "{VCVARS}" >nul 2>&1\n'
                      f'cd /d "{tree}"\n' + body, encoding="utf-8")

    return subprocess.run(["cmd.exe", "/c", str(script)], capture_output=True, timeout=timeout,
                          env=dict(os.environ, MSYS_NO_PATHCONV="1"))


def compiled(tree: pathlib.Path, source: str, timeout: float = 900) -> bool:
    """Whether one scratch translation unit compiles. The negative control of this check.

    This is the control `tools/check_class_combinations.py` makes, at the same strength: an
    instantiation the library refuses must FAIL to compile and one it accepts must PASS, and a
    check that could not fail would be a decoration.
    """
    unit = tree / "control.cpp"
    unit.write_text(source, encoding="utf-8")

    try:
        done = batch(tree, f'cl /nologo /std:c++latest /EHsc /Zs /I include "{unit}" >nul 2>&1\n',
                     timeout)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

    return done.returncode == 0


def control_unit(precision: str, shape: str, pack: str) -> str:
    """A translation unit instantiating one class's entry at one packing axis member."""
    policy = (f"boys::EvalPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kHorner, "
              f"boys::BoysBudget::{BUDGET[precision]}, boys::PackAxis::{pack}, "
              f"boys::FitGranularity::kNarrow, boys::DivisionForm::kRefinedReciprocal, "
              f"boys::RegionBExp::kFast>")
    call = CALL[shape].format(shape=shape[1:], suffix=SUFFIX[precision], policy=policy)

    return ("#include <boys/boys.hpp>\n"
            "void Instantiate() {\n"
            f"    [[maybe_unused]] {SCALAR[precision]} x = {VALUE[precision]};\n"
            f"    [[maybe_unused]] {SCALAR[precision]} xs[8] = {{}};\n"
            f"    [[maybe_unused]] {SCALAR[precision]} out[64] = {{}};\n"
            "    [[maybe_unused]] int ns[8] = {};\n"
            f"    {call}\n"
            "}\n")


def driver_source(classes: list[tuple[str, str, str]]) -> str:
    """The driver, generated: the axes and the figures read from the library's own tables.

    The `DEFAULT` instantiations are the per-class half: `DefaultPolicy<lane, shape, device>` is
    the axis tuple a class's own default resolves to and `DefaultGuarantee` the figure it carries,
    both of which name the class - which is what the per-lane accessor cannot be asked.
    """
    defaults = "\n".join(
        "    Default<boys::Precision::{precision}, boys::Shape::{shape}, "
        "boys::Device::{device}>({index}, \"{label}\");".format(
            index=index, label=f"{precision[1:]} {shape[1:]}", precision=precision, shape=shape,
            device=device)
        for index, (device, precision, shape) in enumerate(classes))

    return f'''#include <boys/boys.hpp>

#include <cstdio>
#include <span>

namespace {{

constexpr char kSep = '\\x1f';

const char* Text(const char* value) {{
    return value == nullptr || value[0] == '\\0' ? "-" : value;
}}

void Figure(const char* kind, const char* label, int lane, int route, int scheme, int axis,
            int partition, int form, int exp, const boys::AccuracyFigure& figure) {{
    std::printf("%s%c%s%c%d%c%d%c%d%c%d%c%d%c%d%c%d%c%d%c%.17g%c%s%c%s\\n",
                kind, kSep, label, kSep, lane, kSep, route, kSep, scheme, kSep, axis, kSep,
                partition, kSep, form, kSep, exp, kSep, figure.available ? 1 : 0, kSep,
                figure.value, kSep, Text(figure.source), kSep, Text(figure.reason));
}}

// The per-class query: the class's own default policy's axes, and the figure that tuple carries.
template <boys::Precision kLane, boys::Shape kShape, boys::Device kDevice>
void Default(int index, const char* label) {{
    using Policy = boys::DefaultPolicy<kLane, kShape, kDevice>;
    const boys::AccuracyFigure figure = boys::DefaultGuarantee<kLane, kShape, kDevice>();

    std::printf("DEFAULT%c%d%c%s%c%d%c%d%c%d%c%d%c%d%c%d%c%d%c%d%c%d%c%d%c%.17g%c%s%c%s\\n",
                kSep, index, kSep, label, kSep, static_cast<int>(kLane), kSep,
                static_cast<int>(kShape), kSep, static_cast<int>(kDevice), kSep,
                static_cast<int>(Policy::kRoute), kSep, static_cast<int>(Policy::kScheme), kSep,
                static_cast<int>(Policy::kPack), kSep, static_cast<int>(Policy::kGranularity),
                kSep, static_cast<int>(Policy::kDivision), kSep,
                static_cast<int>(Policy::kRegionBExp), kSep, figure.available ? 1 : 0, kSep,
                figure.value, kSep, Text(figure.source), kSep, Text(figure.reason));
}}

}} // namespace

int main() {{
    const std::span<const boys::LaneContractInfo> lanes = boys::BoysLaneContracts();

    // Each row carries the lane it describes, and the row's own selector is what every query below
    // is made with: a table ordered differently from the enumeration would answer for one lane
    // while naming another, and that is the defect this prints the pair to catch.
    for (std::size_t i = 0; i < lanes.size(); ++i) {{
        std::printf("LANE%c%zu%c%d%c%s%c%.17g%c%.17g%c%.17g%c%d\\n", kSep, i, kSep,
                    static_cast<int>(lanes[i].precision), kSep, Text(lanes[i].name), kSep,
                    lanes[i].bound, kSep, lanes[i].additive, kSep, lanes[i].plainAdditive, kSep,
                    static_cast<int>(lanes[i].additiveMember));
    }}

    // The routes, as the distinct selector values the route table names: the table holds one row
    // per route and region, so a route serving two regions is one member here and not two.
    int routes = 0;

    for (std::size_t i = 0; i < boys::BoysFitRoutes().size(); ++i) {{
        const boys::FitRouteInfo& row = boys::BoysFitRoutes()[i];
        bool seen = false;

        for (std::size_t j = 0; j < i; ++j) {{
            seen = seen || boys::BoysFitRoutes()[j].route == row.route;
        }}

        if (!seen) {{
            std::printf("AXIS%croute%c%d%c%s\\n", kSep, kSep, static_cast<int>(row.route), kSep,
                        Text(row.name));
            routes += 1;
        }}
    }}

    for (std::size_t i = 0; i < boys::BoysEvalSchemes().size(); ++i) {{
        std::printf("AXIS%cscheme%c%zu%c%s\\n", kSep, kSep, i, kSep,
                    Text(boys::BoysEvalSchemes()[i].name));
    }}
    for (std::size_t i = 0; i < boys::BoysPackAxes().size(); ++i) {{
        std::printf("AXIS%cpacking%c%zu%c%s\\n", kSep, kSep, i, kSep,
                    Text(boys::BoysPackAxes()[i].name));
    }}
    for (std::size_t i = 0; i < boys::BoysFitGranularities().size(); ++i) {{
        std::printf("AXIS%cpartition%c%zu%c%s\\n", kSep, kSep, i, kSep,
                    Text(boys::BoysFitGranularities()[i].name));
    }}
    for (std::size_t i = 0; i < boys::BoysDivisionForms().size(); ++i) {{
        std::printf("AXIS%cdivision%c%zu%c%s\\n", kSep, kSep, i, kSep,
                    Text(boys::BoysDivisionForms()[i].name));
    }}
    for (std::size_t i = 0; i < boys::BoysRegionBExps().size(); ++i) {{
        std::printf("AXIS%cexponential%c%zu%c%s\\n", kSep, kSep, i, kSep,
                    Text(boys::BoysRegionBExps()[i].name));
    }}

    const int schemes = static_cast<int>(boys::BoysEvalSchemes().size());
    const int axes = static_cast<int>(boys::BoysPackAxes().size());
    const int partitions = static_cast<int>(boys::BoysFitGranularities().size());
    const int forms = static_cast<int>(boys::BoysDivisionForms().size());
    const int exps = static_cast<int>(boys::BoysRegionBExps().size());

    for (int l = 0; l < static_cast<int>(lanes.size()); ++l) {{
        for (int r = 0; r < routes; ++r) {{
            for (int s = 0; s < schemes; ++s) {{
                for (int a = 0; a < axes; ++a) {{
                    for (int p = 0; p < partitions; ++p) {{
                        for (int f = 0; f < forms; ++f) {{
                            for (int e = 0; e < exps; ++e) {{
                                Figure("FIG", lanes[l].name,
                                       static_cast<int>(lanes[l].precision), r, s, a, p, f, e,
                                       boys::BoysAccuracyGuaranteed(
                                           lanes[l].precision,
                                           static_cast<boys::FitRoute>(r),
                                           static_cast<boys::EvalScheme>(s),
                                           static_cast<boys::PackAxis>(a),
                                           static_cast<boys::FitGranularity>(p),
                                           static_cast<boys::DivisionForm>(f),
                                           static_cast<boys::RegionBExp>(e)));
                            }}
                        }}
                    }}
                }}
            }}
        }}
    }}

    // The guard. A value one past the last packing axis is not a member of the enumeration and
    // names no combination on any lane, and the accessor must refuse it rather than answer it. A
    // query that answered every tuple with a figure could not fail, and a check resting on one
    // would pass whatever the library did.
    for (int l = 0; l < static_cast<int>(lanes.size()); ++l) {{
        Figure("GUARD", lanes[l].name, static_cast<int>(lanes[l].precision), 0, 0, axes, 0, 0, 0,
               boys::BoysAccuracyGuaranteed(
                   lanes[l].precision, boys::FitRoute::kChebyshev,
                   boys::EvalScheme::kSplitClenshaw, static_cast<boys::PackAxis>(axes),
                   boys::FitGranularity::kCoarsest, boys::DivisionForm::kExactDivision,
                   boys::RegionBExp::kAccurate));
    }}

    // One per class: the bound the class's own default policy carries.
{defaults}

    return 0;
}}
'''


def run_driver(tree: pathlib.Path, classes: list[tuple[str, str, str]], sources: list[str],
               timeout: float) -> str:
    """Compile and run the driver, and answer its stdout.

    The driver is a consumer translation unit and the library is linked beside it from its own
    source list. The compile is the price of reading the library's own answer: `boys.hpp` carries
    the coefficient tables, so every translation unit that includes it pays for them once.
    """
    driver = tree / "check_combination_bounds_driver.cpp"
    driver.write_text(driver_source(classes), encoding="utf-8")

    try:
        built = batch(tree, f'cl /nologo /std:c++latest /EHsc /Od /I include '
                            f'{" ".join(DEFINES)} "{driver}" {chr(32).join(sources)} '
                            f'/Fe:driver.exe\n', timeout)
    except (subprocess.TimeoutExpired, FileNotFoundError) as error:
        raise SystemExit(f"check_combination_bounds: the driver did not build: {error}")

    if built.returncode != 0 or not (tree / "driver.exe").exists():
        sys.stderr.write(built.stdout.decode(errors="replace"))
        sys.stderr.write(built.stderr.decode(errors="replace"))
        raise SystemExit("check_combination_bounds: the driver did not build, so no figure was "
                         "read from the library and this run states nothing about it")

    try:
        ran = subprocess.run([str(tree / "driver.exe")], capture_output=True, text=True,
                             timeout=600)
    except (subprocess.TimeoutExpired, FileNotFoundError) as error:
        raise SystemExit(f"check_combination_bounds: the driver did not run: {error}")

    if ran.returncode != 0:
        raise SystemExit(f"check_combination_bounds: the driver exited {ran.returncode}")

    return ran.stdout


def parse_driver(output: str) -> tuple[dict, dict, dict, dict]:
    """The driver's records: its lanes, axis members, figures and per-class defaults."""
    lanes: dict[int, dict] = {}
    axes: dict[str, list[str]] = {}
    figures: dict[tuple, dict] = {}
    defaults: dict[int, dict] = {}

    for line in output.splitlines():
        fields = line.split(SEP)

        if fields[0] == "LANE":
            lanes[int(fields[1])] = {"lane": int(fields[2]), "name": fields[3],
                                     "bound": float(fields[4]), "additive": float(fields[5]),
                                     "plain": float(fields[6]), "member": int(fields[7])}
        elif fields[0] == "AXIS":
            axes.setdefault(fields[1], []).append(fields[3])
        elif fields[0] in ("FIG", "GUARD"):
            key = (fields[0], fields[1]) + tuple(int(f) for f in fields[2:9])
            figures[key] = {"available": fields[9] == "1", "value": float(fields[10]),
                            "source": fields[11], "reason": fields[12]}
        elif fields[0] == "DEFAULT":
            defaults[int(fields[1])] = {
                "label": fields[2], "lane": int(fields[3]), "shape": int(fields[4]),
                "device": int(fields[5]), "axes": [int(f) for f in fields[6:12]],
                "available": fields[12] == "1", "value": float(fields[13]),
                "source": fields[14], "reason": fields[15]}

    return lanes, axes, figures, defaults


def recorded_cross(run: str) -> dict[tuple[str, str, str, str, str], list[dict]]:
    """The recorded run's cross, keyed by the five axes it names.

    A key can carry more than one row: a lane whose plain reciprocal adds a term beside its base
    states that form's figure as a row of its own. Every row a key carries is kept, because each is
    a bound the run claims.
    """
    start = run.find(CROSS_OPEN)
    end = run.find(CROSS_CLOSE)

    if start < 0 or end < 0 or end < start:
        return {}

    rows: dict[tuple[str, str, str, str, str], list[dict]] = {}

    for match in CROSS_ROW.finditer(run[start:end]):
        key = (match.group("lane"), match.group("route"), match.group("scheme"),
               match.group("partition"), match.group("axis"))
        rows.setdefault(key, []).append({"cells": match.group("cells"),
                                         "outside": match.group("outside"),
                                         "delivered": match.group("delivered"),
                                         "bound": match.group("bound")})

    return rows


def recorded_accessor(run: str) -> list[dict]:
    """The figure the recorded run states the accessor answers, one combination per lane.

    This is the run's own account of the accessor rather than of the code, so a live call to the
    accessor can be held to it: a figure the run recorded and the accessor no longer answers is a
    claim the two do not agree on.
    """
    start = run.find(ACCESSOR_OPEN)

    if start < 0:
        return []

    end = run.find(CROSS_OPEN, start)

    if end < 0:
        return []

    return [match.groupdict() for match in ACCESSOR_ROW.finditer(run[start:end])]


def folded(name: str) -> str:
    """A name with everything but its letters and digits dropped, lowercased.

    The class list names a lane as an enumerator (`kFp32Device`) and the lane table names it as the
    string its rows publish (`fp32-device`); the two are the same lane spelled by two different
    conventions, and this is the spelling that compares them. The caller drops the enumerator's
    leading `k`, which is the naming convention's and no part of the lane's name.
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


def number(text: str) -> float | None:
    """The number a run column states, or None where it states prose instead."""
    try:
        return float(text)
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--revision", default="HEAD",
                        help="the revision whose library is checked (default HEAD)")
    parser.add_argument("--tree", type=pathlib.Path,
                        help="a directory already holding the revision's include/ and src/, "
                             "instead of extracting one")
    parser.add_argument("--no-compile", action="store_true",
                        help="skip the control that proves the one-order exclusion by compiling")
    parser.add_argument("--show", type=int, default=0,
                        help="combination rows to print per class, 0 for the counts alone")
    arguments = parser.parse_args()

    revision = (git("rev-parse", "--short", arguments.revision).stdout.strip()
                or arguments.revision)
    scratch = None

    if arguments.tree is not None:
        tree = arguments.tree.resolve()
    else:
        scratch = pathlib.Path(tempfile.mkdtemp(prefix="boys-bounds-"))
        tree = scratch
        materialize(arguments.revision, tree)

    try:
        header_path = tree / "include" / "boys" / "boys.hpp"

        if not header_path.exists():
            print(f"check_combination_bounds: no header at {header_path}; the library was not "
                  "read, so this run states nothing about it")
            return 1

        header = header_path.read_text(encoding="utf-8", errors="replace")
        run_path = tree / "tests" / "data" / "boys_accuracy_gate_run.txt"
        run = run_path.read_text(encoding="utf-8", errors="replace") if run_path.exists() else ""

        classes = class_rows(header)
        one_order = one_order_shapes(header)
        lane_names = [member for member, _ in enumerators(header, "Precision")]
        shape_names = [member for member, _ in enumerators(header, "Shape")]

        if not classes or not one_order or not lane_names or not shape_names:
            print("check_combination_bounds: the classes or the shapes were read as empty, so this "
                  "run states nothing about the library and its exit status is a failure rather "
                  "than a clean bill")
            return 1

        # The control: a one-order shape cannot take the orders packing axis. One instantiation
        # each way, for every one-order shape the library's own documentation names.
        if not arguments.no_compile:
            for shape in sorted(one_order):
                at_orders = compiled(tree, control_unit(CONTROL_LANE, shape, ORDERS_AXIS))
                off_orders = compiled(tree, control_unit(CONTROL_LANE, shape, "kArguments"))

                if at_orders or not off_orders:
                    print(f"check_combination_bounds: [{shape}] control: at the orders axis "
                          f"compiles = {at_orders}, without it compiles = {off_orders} - the "
                          "exclusion this check rests on does not hold, so its counts are not "
                          "admissible")
                    return 1

                print(f"check_combination_bounds: [{shape}] control: the orders axis is refused by "
                      "the compiler and the arguments axis is accepted")

        sources = library_sources(tree)
        output = run_driver(tree, classes, sources, timeout=3600)
        lanes, axes, figures, defaults = parse_driver(output)

        if not lanes or not axes or not figures:
            print("check_combination_bounds: the driver answered no lane, no axis or no figure, so "
                  "this run states nothing about the library")
            return 1

        route_members = axes.get("route", [])
        scheme_members = axes.get("scheme", [])
        pack_members = axes.get("packing", [])
        partition_members = axes.get("partition", [])
        form_members = axes.get("division", [])
        exp_members = axes.get("exponential", [])

        for name, members in (("route", route_members), ("scheme", scheme_members),
                              ("packing", pack_members), ("partition", partition_members),
                              ("division", form_members), ("exponential", exp_members)):
            if not members:
                print(f"check_combination_bounds: the library's own table names no {name} member, "
                      "so no combination can be enumerated")
                return 1

        cross = recorded_cross(run)

        # A class names its lane by enumerator, and every figure this check reports for the class
        # was read at a row of the lane table: the two are tied together here rather than assumed.
        # The row at the enumerator's own position must carry that position as its own selector and
        # a name folding to the enumerator's, or the name-keyed lookups below answer for one lane
        # while the row they are printed against names another.
        lane_of: dict[str, int] = {}

        for member, _ in enumerators(header, "Precision"):
            position = lane_names.index(member)
            row = lanes.get(position)

            if row is None or row["lane"] != position or folded(row["name"]) != folded(member[1:]):
                print(f"check_combination_bounds: the lane table's row {position} carries selector "
                      f"{None if row is None else row['lane']} and name "
                      f"{None if row is None else row['name']!r}; the enumerator at that position "
                      f"is {member}. The table and the enumeration are not in the same order, so a "
                      "figure read at a row cannot be reported against the class that named it")
                return 1

            lane_of[member] = position

        unnamed = sorted({precision for _, precision, _ in classes
                          if precision not in lane_of})

        if unnamed:
            print("check_combination_bounds: the class list names lane(s) the Precision enumeration "
                  "does not declare: " + ", ".join(unnamed))
            return 1

        print(f"revision {revision}: {len(classes)} class(es) = "
              f"{sum(1 for c in classes if c[0] == 'kHost')} host + "
              f"{sum(1 for c in classes if c[0] == 'kDevice')} device")
        print("  lanes, the library's own rows: "
              + ", ".join(lanes[i]["name"] for i in sorted(lanes)))
        print(f"  axes, the library's own tables: {len(route_members)} route(s) "
              f"({', '.join(route_members)}) | {len(scheme_members)} scheme(s) "
              f"({', '.join(scheme_members)}) | {len(pack_members)} packing "
              f"({', '.join(pack_members)}) | {len(partition_members)} partition(s) "
              f"({', '.join(partition_members)}) | {len(form_members)} division "
              f"({', '.join(form_members)}) | {len(exp_members)} exponential(s) "
              f"({', '.join(exp_members)})")
        print("  one-order shapes, read from the Shape enumerators' own documentation: "
              + ", ".join(sorted(one_order)))
        print("  the recorded run: "
              + (f"read, {len(cross)} key(s)" if cross else "NOT READ"))

        # A run key names a lane, a route, a scheme, a partition and an axis; each must be a name
        # the library's own tables state, or the run and this check are not talking about the same
        # combination and neither the coverage nor the honour figure means what it says.
        library_names = {
            "lane": {lanes[i]["name"] for i in lanes},
            "route": set(route_members) | {"default"},
            "scheme": set(scheme_members),
            "partition": set(partition_members),
            "axis": set(pack_members) | {"off"},
        }
        unmatched = sorted({f"{field}={value}"
                            for key in cross
                            for field, value in zip(("lane", "route", "scheme", "partition",
                                                     "axis"), key)
                            if value not in library_names[field]})

        if unmatched:
            print("  the recorded run names component(s) the library's tables do not, so those "
                  "rows are not these combinations: " + ", ".join(unmatched))

        total = 0
        bounded = 0
        unbounded: list[tuple] = []

        print("\n  class                                  possible  bounded  no bound  lane")
        print("  " + "-" * 88)

        for index, (device, precision, shape) in enumerate(classes):
            lane_index = lane_of[precision]
            # The lane table's own selector, which is what the driver's records carry in their lane
            # field: the two are tied together above, and the keys are built from the row so that
            # they are right by construction rather than by that tie holding.
            selector = lanes[lane_index]["lane"]
            # A one-order shape drops the orders member of the packing axis: one order has one
            # order to put in a vector lane, so the axis has one member there and not two.
            packs = ([m for m in pack_members if m != "orders"]
                     if shape in one_order else pack_members)
            class_total = 0
            class_unbounded = 0
            # The figures the accessor answers this class's carried combinations with, by the two
            # axes that move a figure. The other four axes enter the answer through carried or
            # refused alone, and that is a reading of this map rather than a statement beside it.
            by_axis_pair: dict[tuple[str, str], list[float]] = {}

            for route, scheme, pack, partition, form, exp in itertools.product(
                    route_members, scheme_members, packs, partition_members, form_members,
                    exp_members):
                key = ("FIG", lanes[lane_index]["name"], selector,
                       route_members.index(route), scheme_members.index(scheme),
                       pack_members.index(pack), partition_members.index(partition),
                       form_members.index(form), exp_members.index(exp))
                figure = figures.get(key)
                class_total += 1
                total += 1

                if figure is None:
                    print(f"check_combination_bounds: the driver answered no figure for {key} - "
                          "the table is incomplete")
                    return 1

                if figure["available"]:
                    bounded += 1
                    values = by_axis_pair.setdefault((form, exp), [])
                    if figure["value"] not in values:
                        values.append(figure["value"])
                else:
                    class_unbounded += 1
                    unbounded.append((precision, shape, route, scheme, pack, partition, form,
                                      exp, figure["reason"]))

            default = defaults[index]
            default_axes = "/".join([route_members[default["axes"][0]],
                                     scheme_members[default["axes"][1]],
                                     pack_members[default["axes"][2]],
                                     partition_members[default["axes"][3]],
                                     form_members[default["axes"][4]],
                                     exp_members[default["axes"][5]]])

            print(f"  {precision[1:] + ' ' + shape[1:]:<38} {class_total:>8}  "
                  f"{class_total - class_unbounded:>7}  {class_unbounded:>8}  "
                  f"{lanes[lane_index]['name']}")

            # The per-class half, and the sentence the brief asks for on every row: the figure is
            # the lane's, and the shape reaches it by choosing the tuple and in no other way.
            print(f"      its own default policy: {default_axes} at bound "
                  + (f"{default['value']:.6g}" if default["available"] else "NONE"))
            print(f"      DefaultGuarantee<{precision}, {shape}, {device}> - the answer is "
                  f"per-lane: the accessor takes no Shape, so this class's figure is the one every "
                  f"{lanes[lane_index]['name']} class reads at these axes")

            # Which axes the figure moves with, read from the table rather than said beside it:
            # every combination carried at one (division form, exponential) pair is answered with
            # one figure, so the four axes this class enumerates beside those two change the answer
            # by admitting or refusing a combination and in no other way.
            if by_axis_pair:
                print("      its figures, by division form and exponential: "
                      + " | ".join(f"{form}/{exp} " + ", ".join(f"{value:.6g}" for value in values)
                                   for (form, exp), values in sorted(by_axis_pair.items()))
                      + " - each pair answers one figure for every combination it carries, so the "
                      "route, the scheme, the packing axis and the partition move none: they only "
                      "admit or refuse")

            if default["available"] and default["source"] not in ("-", ""):
                print(f"      stated as: {default['source']}")

            if not default["available"]:
                print(f"      refused: {default['reason']}")

            if arguments.show:
                for route, scheme, pack, partition, form, exp in itertools.islice(
                        itertools.product(route_members, scheme_members, packs, partition_members,
                                          form_members, exp_members), arguments.show):
                    key = ("FIG", lanes[lane_index]["name"], selector,
                           route_members.index(route), scheme_members.index(scheme),
                           pack_members.index(pack), partition_members.index(partition),
                           form_members.index(form), exp_members.index(exp))
                    figure = figures[key]
                    state = f"{figure['value']:.6g}" if figure["available"] else "NO BOUND"
                    note = figure["source"] if figure["available"] else figure["reason"]
                    print(f"      {route:<17} {scheme:<15} {pack:<10} {partition:<9} "
                          f"{form:<18} {exp:<9} {state:>12}  {note}")

        # The guard: a value one past the last packing axis names no combination, and a query that
        # answered it with a figure would be one that cannot fail.
        guard_ok = 0

        for lane_index in sorted(lanes):
            figure = figures.get(("GUARD", lanes[lane_index]["name"], lanes[lane_index]["lane"], 0, 0,
                                  len(pack_members), 0, 0, 0))

            if figure is not None and not figure["available"]:
                guard_ok += 1

        print(f"\n  the guard, one past the last packing axis: refused with a reason on {guard_ok} "
              f"of {len(lanes)} lane(s)")

        if guard_ok != len(lanes):
            print("check_combination_bounds: the accessor answered a value outside the enumeration "
                  "with a figure, or the guard could not be read: this check cannot tell a stated "
                  "bound from an unstated one and its answer is not admissible")
            return 1

        if bounded == 0:
            print("check_combination_bounds: no combination anywhere was answered with a figure, so "
                  "a query refusing everything would pass this check and it states nothing")
            return 1

        # The recorded run's half. It covers the five axes it names and reads every cell of that
        # cross at each division form; it covers the region-B exponential nowhere.
        uncovered = 0
        dishonoured: list[tuple] = []

        for device, precision, shape in classes:
            lane_index = lane_of[precision]
            lane_name = lanes[lane_index]["name"]

            for route, scheme, pack, partition in itertools.product(
                    route_members, scheme_members, pack_members, partition_members):
                if shape in one_order and pack == "orders":
                    continue

                key = (lane_name, route, scheme, partition, pack)
                rows = cross.get(key)

                if not rows:
                    uncovered += 1
                    continue

                for row in rows:
                    outside = number(row["outside"])
                    bound = number(row["bound"])
                    delivered = number(row["delivered"])

                    if outside is not None and outside > 0.0:
                        dishonoured.append((key, f"{row['outside']} cell(s) outside", row["bound"]))

                    if bound is not None and delivered is not None and delivered > bound:
                        dishonoured.append((key, f"delivered {row['delivered']}", row["bound"]))

        print(f"    the run's cross is keyed by (lane, route, scheme, partition, packing axis) and "
              f"states {len(cross)} key(s): {total - uncovered} of the {total} class-combination "
              f"cells project onto one, {uncovered} do not")
        print(f"    the axes no row of the run is keyed by: the division form, whose "
              f"{len(form_members)} members each row's counts are taken over together, and the "
              f"region-B exponential, which has no column at all - so every cell's figure under "
              "those two axes is reported here as unchecked against measurement rather than "
              "passed, and the exp axis in particular is covered by no row of the run")
        print("    the run names no shape either, so on one lane its rows are the evidence for "
              "every class at once: the measurement behind a figure cannot distinguish the classes "
              "that share a lane, and the only per-class statement about a bound is the axis tuple "
              "each class's own default policy resolves to, printed above")

        # The bar a row of the run was judged against and the figure the accessor answers for that
        # combination are two statements, and on some lanes they are not the same number: a lane
        # that states the plain reciprocal's figure apart has its cells judged against the widest of
        # the figures it states, and the half lanes' rows state a figure their own sentence claims
        # only where the value exceeds the format's half digit, which is a term of the value and not
        # of the call. Neither is a failure on its own. It is counted and printed because a reader
        # taking the honour comparison below for a comparison against the accessor's own figure
        # would be reading one number where the library states two, on exactly those lanes.
        bars: dict[str, list[int]] = {}

        for (lane_name, route, scheme, partition, axis), rows in cross.items():
            lane_index = next((i for i in lanes if lanes[i]["name"] == lane_name), None)

            if (lane_index is None or route not in route_members or scheme not in scheme_members
                    or axis not in pack_members or partition not in partition_members):
                continue

            answered = set()

            for form in form_members:
                for exp in exp_members:
                    figure = figures.get(("FIG", lane_name, lanes[lane_index]["lane"],
                                          route_members.index(route),
                                          scheme_members.index(scheme),
                                          pack_members.index(axis),
                                          partition_members.index(partition),
                                          form_members.index(form), exp_members.index(exp)))

                    if figure is not None and figure["available"]:
                        answered.add(figure["value"])

            seen = bars.setdefault(lane_name, [0, 0])

            for row in rows:
                bar = number(row["bound"])
                seen[1] += 1

                if bar is not None and bar not in answered:
                    seen[0] += 1

        parted = [f"{lane} {count} of {total_rows}"
                  for lane, (count, total_rows) in sorted(bars.items()) if count]
        flagged = sum(count for count, _ in bars.values())
        swept = sum(total_rows for _, total_rows in bars.values())

        print(f"    the run's bar against the accessor's figure for the same combination: "
              f"{flagged} of {swept} row(s) carry a bar no figure the accessor answers"
              + (f" ({', '.join(parted)})" if parted else "")
              + " - the bar is the statement the gate judged its own cells against and the figure "
                "is the lane's contract, so each is held to its own; the places the two part are "
                "printed here so that neither is read for the other")

        if dishonoured:
            print(f"\n{len(dishonoured)} row(s) of the recorded run carry a figure the code does "
                  "not keep:")
            for key, seen, bound in dishonoured:
                print(f"  {', '.join(key)}: {seen}, bound {bound}")
            print("A bound the library states and its own code leaves is the defect this check "
                  "exists for.")
            return 1

        # The run's own account of the accessor, held to the live one: one combination per lane,
        # with the figure the run recorded the accessor answering for it. A figure that no form or
        # exponential of that combination answers any more is a claim the record and the library
        # do not agree on. The figures are compared as the run prints them, three significant
        # digits, so the two are read at the same precision.
        accessor_rows = recorded_accessor(run)
        disagreed: list[tuple] = []

        for row in accessor_rows:
            lane_index = next((i for i in lanes if lanes[i]["name"] == row["lane"]), None)

            if (lane_index is None or row["route"] not in route_members
                    or row["scheme"] not in scheme_members or row["axis"] not in pack_members
                    or row["partition"] not in partition_members):
                continue

            stated = set()

            for form in form_members:
                for exp in exp_members:
                    figure = figures.get(("FIG", row["lane"], lanes[lane_index]["lane"],
                                          route_members.index(row["route"]),
                                          scheme_members.index(row["scheme"]),
                                          pack_members.index(row["axis"]),
                                          partition_members.index(row["partition"]),
                                          form_members.index(form), exp_members.index(exp)))

                    if figure is not None and figure["available"]:
                        stated.add(f"{figure['value']:.3g}")

            if stated and row["bound"] not in stated:
                disagreed.append((row, sorted(stated)))

        print(f"    the run's accessor table: {len(accessor_rows)} lane(s) with a recorded figure, "
              f"{len(disagreed)} of them a figure the live accessor does not answer")

        if disagreed:
            print(f"\n{len(disagreed)} lane(s) whose recorded figure the accessor does not answer "
                  "at any form or exponential of that combination:")
            for row, stated in disagreed:
                print(f"  {row['lane']}, {row['route']}, {row['scheme']}, {row['partition']}, "
                      f"{row['axis']}: the run states {row['bound']}, the accessor answers "
                      f"{', '.join(stated)}")
            print("A figure the record states and the library no longer answers is a claim the two "
                  "do not agree on.")
            return 1

        print(f"\n  combinations checked: {total}")
        print(f"  combinations with a stated bound: {bounded}")
        print(f"  combinations with NO bound: {len(unbounded)}")

        if unbounded:
            print(f"\n{len(unbounded)} combination(s) have no bound at all:")
            for row in unbounded:
                precision, shape, route, scheme, pack, partition, form, exp, reason = row
                print(f"  {precision} {shape} {route}/{scheme}/{pack}/{partition}/{form}/{exp}: "
                      f"{reason}")
            print("A combination nothing bounds is one no caller can rely on.")
            return 1

        if total == 0:
            print("\nno class was checked: the class list or the axes were read as empty, so this "
                  "run states nothing about the library and its exit status is a failure rather "
                  "than a clean bill")
            return 1

        print("\nevery class's every fundamentally-possible combination has a stated bound, and "
              "every combination the recorded run covers keeps it")
        return 0
    finally:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
