#!/usr/bin/env python3
"""Which combinations the accuracy gate's recorded run measures, against the ones that exist.

The condition has a "measured (bounds)" leg: a combination the library offers is
supposed to have its bound measured and published. Nothing today says whether the gate
covers a particular combination, so a combination can be read as measured when the gate
has never looked at it. This check is that statement.

WHAT IS COMPARED, AND WHERE EACH SIDE COMES FROM

  * the combinations that exist are the axes of the library's own guarantee,
    ``BoysAccuracyGuaranteed(precision, route, scheme, axis, granularity, form, exp)``
    (include/boys/boys.hpp). Every axis member is read from the table the library
    publishes for it - ``BoysFitRoutes``/``BoysFitRoutesF32``, ``BoysEvalSchemes``,
    ``BoysPackAxes``, ``BoysFitGranularities``, ``BoysDivisionForms``, ``BoysRegionBExps``
    and ``BoysLaneContracts`` - and never from a list written here. The tables' own sizes
    are read too, so a member added to an axis and given a row moves this check's universe
    rather than leaving it behind;

  * the classes are the (device, precision, shape) triples the library's default-policy
    list names (include/boys/boys_build_defaults.hpp, ``BOYS_BUILD_DEFAULT_ROWS``), which
    is the library's own statement of one row per class;

  * what the run measures is the combination book of the recorded run: the rows under
    ``combination ... cells outside delivered bound state``, each one a (lane, route,
    scheme, partition, axis) the gate measured over the committed grid;

  * which class each row measured is read from the gate and from the library: the
    entries the combination block instantiates are read out of the block itself
    (tests/boys_accuracy_gate.cpp, between its ``// ---- the combinations`` marker and
    the next ``// ---- `` marker), and each entry's own class is read from the
    ``DefaultPolicy<Precision::..., Shape::...>`` its declaration defaults to - the
    library's own statement of what question that entry answers. A class whose shape is
    not one of those is not measured by this book, whatever its lane's row says. The
    entries are read per machine - the block's host entries and its device arms - and a
    class is answered by the entries of its own machine: a device class is not measured
    through a host entry, which is the other machine's, and a device lane's rows reach
    Shape::kAllOrders alone where its class list also names kSingle and kAllN. Those two
    classes are counted as measured through another class's entry, and the count beside them
    is a count of cells an entry of another shape produced.

THE THREE GRANULARITIES, AND WHERE THEY DO NOT MATCH

The question is per class and per combination. The three granularities this
check has to cross are not the same, and saying so is half of what it prints:

  * the accessor answers per LANE. It takes no Shape: its doc says the bound "is the
    lane's and not the axes'", and a class is a (device, precision, shape) triple. So
    the figure a class is answered with is its lane's figure, and one measurement of a
    lane's figure answers that figure for every class on the lane. That is the library's
    design and not an omission here - but it means the *figure* a class holds is not the
    same claim as a *measurement of that class's entry*, which is what the gate's
    combination book performs;

  * the gate's combination book is keyed by five axes - lane, route, scheme, partition,
    axis - and measures each of its rows through ONE entry per lane. Its rows therefore
    name a lane and not a class, and a reader comparing a class against this table has to
    know which shape produced the figure;

  * the gate measures the division form inside each row (its arms read every cell at
    every form the library carries) and does not cross the region-B exponential at all.
    The accessor takes that member as an argument, so a member the run never named is a
    combination the library carries and the gate does not measure.

WHICH REGION-B MEMBER A ROW WAS READ AT

A member is not one kind of thing across the lanes, and reading it as one is how a row
gets credited at a member nothing read:

  * on a host lane the member is a policy argument of the call, the run's own member
    sentence states which members its cells were read at, and the sentence's own scope word
    and count are what this check reads. A sentence scoped to other cells than the host
    lanes' sentence is not that statement, and this check reports the scope word rather
    than reading the count off it;

  * on a device lane the member is not an argument at all: the launched entries are
    distinct functions, and the one statement of which member an entry runs is the device
    option table (``BoysDeviceOptions()``, src/boys_cuda.cpp), reached through the
    enumerator's own documentation of the C++ name it is
    (include/boys/boys_cuda_options.hpp). So a device lane's cells are counted at the
    members the entries the gate's block arms that lane with run. They are *not* counted at
    the member the lane's contract row publishes its term beside the base under: that
    member is where the term belongs and not where the entry is, and on fp32-device the two
    differ - the row's 8e-8 term is under the fast member, 2.3e-07, and the launched
    all-orders entry the arms name runs the accurate one, 1.5e-07. A credit read off the row
    would name a member no arm read, which is the false credit this read exists to close.

    Three reads can come back empty and each is a finding rather than a credit: an arm
    naming an entry the table does not carry; an arm naming one the table carries at two
    members while the arm's own spelling names neither - the bare name is the entry's
    default, which the build states as a seam and not as a value, so the cell is
    not credited at either member; and a machine the class list names that the block arms
    with no entry at all. The member the table carries for a name is read once per name, and
    a lane is credited only for the arms the block names for it.

WHAT "THE LIBRARY CARRIES" MEANS, AND WHY IT IS READ RATHER THAN TAKEN

The universe this check measures against is not the cross of the tables on trust. A tuple
of the accessor's seven axes is carried when ``BoysAccuracyGuaranteed`` answers
``available = true`` for it, and the accessor's answer is read from the revision's own
source (src/boys.cpp):

  * the accessor asks a ``Carries*`` rule, and the rules' parameters are the axes they can
    see. Each rule's only refusal is a guard on the enumerations, so a tuple built out of
    the tables is carried - and a rule that could refuse one is a finding, because then
    the width of the gap this check prints would be the cross's and not the library's;

  * the double lane's carriage is decided by the partition rows' own coverage of the
    routes and the packing axes (``FitGranularityHasRoute``/``FitGranularityHasAxis``), so
    the rows' ``routes`` and ``axes`` fields are read too: a row that leaves a member
    clear refuses a tuple that is inside every enumeration.

Both reads are printed with what they found, so a universe read off a rule this check
misunderstood is visible in its own output rather than absorbed into a total.

WHAT IT READS AND WHAT IT DOES NOT

The library's facts are read at the revision the recorded run names, through
``git show <revision>:<path>`` - never out of the working tree, which other work may hold
open - so the constants this check compares against are the ones the recorded run was
produced from. The gate's own source is read from the working tree, which is where the
gate this run records lives; a gate edited since the run is a finding the run's own
revision line carries and ``tools/check_recorded_run.py`` is the check for it.

The one-order shapes' exclusion of the orders packing axis is not assumed. ``Shape::kSingle``
and ``Shape::kFixedN`` have one order to put in a lane, so the axis that fills a lane with a
ladder's orders has nothing to fill it with; but a check that could not fail is a decoration,
so this one compiles an instantiation of each one-order shape at that member and requires the
compiler to refuse it, and one without that member and requires the compiler to accept it.

That control is off by default and turned on with ``--compile``, because it is not cheap: a
syntax-only instantiation of this library's header costs minutes of one core on the machine this
was written on, which is a price the default path of a table read should not pay while other work
runs on the same host. The default path says in its own output which of the two it did, so a run
that asserted the exclusion never reads as one that proved it.

The control compiles against the revision's own headers, materialised into a scratch directory
from the revision's tree, and not against the working tree: a header in the working tree may have
a writer in it, and a compile against that tree is a claim about whatever the writer had reached.

Usage:
    python tools/check_gate_covers_combinations.py
    python tools/check_gate_covers_combinations.py --full
    python tools/check_gate_covers_combinations.py --compile

`--check` is accepted for the invocation the workflow uses uniformly. It is not a mode: this
tool has no other one, so a gap exits non-zero with or without it.

Exit status is 0 only when every combination the library carries is measured by the recorded
run, and 1 otherwise - when one is unmeasured, when the run names no revision, when a
revision cannot be resolved, when this check's own read came back empty, or when a read the
counts rest on did not hold: a carriage rule that can refuse a tuple inside the enumerations,
a partition row that leaves a route or an axis clear, an accessor whose answers could come
from a guard this check has not read, a form or member statement the run does not carry or
carries at another count than the library's, that member statement scoped to cells this check
cannot tie to the host lanes, a device arm naming an entry the device option table does not
carry, a machine whose classes the class list names and the block arms with no entry, or an
include tree that could not be rebuilt for the compile control. A run that finds nothing
exits 1: three tools in this tree reported success over a broken read in one night, so a
silent zero must never read as a pass.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
RUN = REPO / "tests" / "data" / "boys_accuracy_gate_run.txt"
GATE = REPO / "tests" / "boys_accuracy_gate.cpp"

# The headers a library fact is read from. `boys.hpp` is on this list for what it
# declares - the Shape and Precision enumerations and every entry's own class - and the
# other two for the axes they own; `boys_build_defaults.hpp` for the class list, and
# `src/boys.cpp` for the tables and the figures the accessor answers from. The last two
# are the device surface, read for the one fact the device lanes need and no other table
# carries: the region-B member each launched entry runs, which is the entry's own and not
# a policy argument, so a lane's cells are read at it whatever the lane's row states
# beside its base. The device-callable entries are rows of the same table; this check
# reads them for the same reason and finds no arm of the block's naming one.
LIBRARY = (
    "include/boys/boys.hpp",
    "include/boys/accuracy.hpp",
    "include/boys/backend.hpp",
    "include/boys/boys_build_defaults.hpp",
    "include/boys/boys_cuda_options.hpp",
    "src/boys.cpp",
    "src/boys_cuda.cpp",
)

REVISION = re.compile(r"^accuracy gate, revision[ \t]+(?P<rev>\S+)[ \t]*$", re.M)

# What the run's build reached: `device lanes : 4 of 4 measured on a card`. A run
# whose first number is short of its second measured part of the device surface,
# and the combinations it leaves uncovered are on the lanes it did not reach. That
# shortfall is the build's, so the verdict below names the record and the remedy
# rather than reporting the library as carrying what no row measures.
DEVICE_LANES = re.compile(r"^device lanes : (?P<measured>\d+) of (?P<carried>\d+) "
                          r"measured on a card[ \t]*$", re.M)

# The gate's combination book: its marker comment opens it and the next section marker
# closes it. Both are the gate's own, so the block this check reads is the block the gate
# says is the combination book.
BLOCK_OPEN = re.compile(r"^[ \t]*// ---- the combinations:", re.M)

# The combination book is not one section: the device lanes' arms and the double and
# half arms are sections of their own inside it, and the book ends where the gate opens
# the next book. The gate's own phrase for that boundary is a book "counted apart" - a
# book whose numbers are held away from the arithmetic above - so the search takes the
# first section marker that says so rather than the first marker after this one, which
# would stop the read at the first of the book's own sub-sections.
BLOCK_CLOSE = re.compile(r"^[ \t]*// ---- [^\n]*\bcounted apart\b", re.M)
BLOCK_NEXT = re.compile(r"^[ \t]*// ---- ", re.M)

# The combination book's header, which names its six columns and is what makes the rows
# below it combination rows rather than rows of the accessor table that follows (whose
# first five columns are the same five axis names).
ROW_HEADER = re.compile(r"^\s*combination\s+cells\s+outside\s+delivered\s+bound\s+state\s*$", re.M)
DASHES = re.compile(r"^\s*-{40,}\s*$", re.M)

# A row of the combination book: five comma-separated axis members (lane, route, scheme,
# partition, packing axis), then the cell count, the count of cells outside the bound, the
# delivered figure, the bound the row is judged by, and the state. The axis members carry
# no space, so the first numeric column is what ends them, and the count of cells is an
# integer where every other book's row may carry `n/a` there.
COMBINATION_ROW = re.compile(
    r"^[ \t]*(?P<axes>(?:[^,\n]+,[ \t]*){4}[^,\n]+?)[ \t]+(?P<cells>\d+)[ \t]+(?P<over>\d+)[ \t]+"
    r"(?P<delivered>\S+)[ \t]+(?P<bound>\S+)[ \t]+(?P<state>\S.*?)[ \t]*$", re.M)

# Every column separator in this file's two row patterns is horizontal whitespace, never
# `\s`: a `\s` separator reaches across a line break, which turns the dashes line above a
# table into a row whose columns are the line below it and silently drops that line.

# The entry book, which is the run's only book keyed by the entry a caller names rather
# than by the lane. Its marker opens it in the gate and in the run, and its own result
# line closes it in the run.
ENTRY_BOOK_MARKER = re.compile(r"^[ \t]*(?://\s*)?---- the entry book:", re.M)
ENTRY_BOOK_OPEN = re.compile(r"^the entry book:", re.M)
ENTRY_BOOK_CLOSE = re.compile(r"^\s*ENTRY RESULT:", re.M)

# The entries the entry book declares, in the gate's own comment: the label it prints,
# then the entry the label reaches.
ENTRY_BOOK_ENTRY = re.compile(r"^[ \t]*//[ \t]+(?P<label>[a-z][a-z ,\-]*?)[ \t]{2,}"
                              r"(?P<text>\S[^\n]*?)[ \t]*$", re.M)

# A row of the entry book: the entry's name (which contains spaces), the route, the
# scheme, the packing axis, then the figures the row carries and its state. Its columns
# are separated by runs of two or more spaces - the first column is padded to the longest
# entry's name - and the state is what follows the last run, so a row is placed by its own
# columns rather than by a state vocabulary this file would have to spell. The state's own
# meaning is read the same way: a row whose cell count is an integer measured something,
# and a row whose figures are `n/a` did not.
ENTRY_ROW = re.compile(r"^ {2}(?P<entry>\S[^\n]*?)[ \t]{2,}(?P<rest>\S[^\n]*?)[ \t]*$", re.M)

# The run's columns are padded, so a run of two or more spaces separates the entry's name
# from what follows and the last of them separates the state; the route, scheme, axis and
# cell count are the first four whitespace-separated fields between them, which is what
# the padding cannot merge because each of those four is a single token.
ENTRY_COLUMNS = re.compile(r"[ \t]{2,}")

# An enum member with its documentation after `///<`, which is where the Shape
# enumeration states which shapes evaluate one order. Only the first member of an
# enumeration carries an initialiser, so it is optional here.
ENUM_MEMBER = re.compile(r"^[ \t]*(k[A-Za-z0-9_]+)[ \t]*(?:=[^,]*)?,[ \t]*(?:///<[ \t]*(.*))?$", re.M)
ONE_ORDER = re.compile(r"\bone order\b", re.I)

# A name function: `case EvalScheme::kHorner: return "horner";`. The library spells each
# axis member once, here, and the tables read these names rather than repeating them.
NAME_CASE = re.compile(r"case\s+(?P<enum>[A-Za-z_][A-Za-z0-9_]*)::(?P<member>k[A-Za-z0-9_]+)\s*:"
                       r"\s*return\s+\"(?P<name>[^\"]*)\"\s*;")

# The enumeration each table is a table of, and the member name a table's own rows use.
# Read by the accessor that publishes the table, so the members a table carries are the
# members its rows name and not another table's.
TABLE_ENUM = {
    "BoysFitRoutes": ("FitRoute", "route"),
    "BoysFitRoutesF32": ("FitRoute", "route"),
    "BoysEvalSchemes": ("EvalScheme", "scheme"),
    "BoysPackAxes": ("PackAxis", "axis"),
    "BoysFitGranularities": ("FitGranularity", "granularity"),
    "BoysDivisionForms": ("DivisionForm", "form"),
    "BoysRegionBExps": ("RegionBExp", "exp"),
    "BoysLaneContracts": ("Precision", "lane"),
}

# One row of ``BoysFitRoutes()``/``BoysFitRoutesF32()``: the route's enumerator beside
# the name the run prints. The row's later fields are figures and are not read here.
ROUTE_ROW = re.compile(r"\{\s*FitRoute::(?P<member>k[A-Za-z0-9_]+)\s*,\s*"
                       r"\"(?P<name>[^\"]*)\"")

# The device surface's own table, ``kDeviceOptions`` in src/boys_cuda.cpp: one row per
# DeviceEntry, in the enumerator's order. It is where the member a launched entry runs is
# stated, and that member is not a policy argument of the call - it is the entry.
DEVICE_OPTION_TABLE = "kDeviceOptions[]"

# The entry a device row is, and the region-B member it runs. Both fields are read from the
# row's own text rather than by position, so a field added to DeviceOptionInfo moves no
# reading here.
DEVICE_OPTION_ENTRY = re.compile(r"DeviceEntry::(k[A-Za-z0-9_]+)")
DEVICE_OPTION_MEMBER = re.compile(r"RegionBExp::(k[A-Za-z0-9_]+)")

# The C++ entry a DeviceEntry enumerator is, which the library states in the enumerator's
# own documentation: `kAllOrdersF32, ///< BoysCuda::AllOrdersF32, launched`. One entry may
# be two enumerators - the f32 single entry is one name at two members, a row each - so a
# name is read to a SET of members, and the caller treats a set of more than one as a name
# whose member the spelling does not state rather than as a name reaching both.
DEVICE_ENTRY_NAME = re.compile(r"\bBoysCuda::([A-Za-z0-9_]+)")

# The carriage rules the accessor's answer rests on. Each is a function whose parameters are
# the axes it can see, so a rule that does not take the division form or the region-B
# exponential cannot refuse on either - which is the statement that makes every (form, exp)
# pair carried wherever the tuple under it is.
CARRIER = re.compile(r"Carriage\s+(?P<name>Carries[A-Za-z0-9_]*)\s*\((?P<params>[^)]*)\)")

# The double lane's carriage adds one gate the other lanes' rules do not make: the partition
# row's own coverage. Both fields are set once per row of ``BoysFitGranularities()``, and the
# rows are built rather than listed, so this is where that table's members are read from.
PARTITION_ROUTES = re.compile(r"built\[(?P<row>\d+)\]\.routes\s*=\s*(?P<value>[^;]+);")
PARTITION_AXES = re.compile(r"built\[(?P<row>\d+)\]\.axes\s*=\s*(?P<value>[^;]+);")

# A partition row's own field, as the table's rows are numbered by the index they are built at.
PARTITION_ROW_INDEX = re.compile(r"built\[(?P<row>\d+)\]\.granularity\s*=\s*"
                                 r"FitGranularity::(?P<member>k[A-Za-z0-9_]+)\s*;")

# The name a partition row's coverage field is written with, and what the name is defined as.
# A row that names a symbol is read through the symbol's own definition rather than through
# the symbol's spelling, so a bitmask that stops covering an axis is read as that.
BIT_DEFINITION = re.compile(r"constexpr\s+unsigned\s+(?P<name>[A-Za-z0-9_]+)\s*=\s*(?P<value>[^;]+);")

# An entry's declaration: its default policy names the class the entry answers for -
# `DefaultPolicy<Precision::kFp64, Shape::kAllOrders>` - and that is the library's own
# statement of what question the entry answers.
ENTRY_CLASS = re.compile(r"DefaultPolicy<\s*Precision::(?P<precision>k[A-Za-z0-9_]+)\s*,\s*"
                         r"Shape::(?P<shape>k[A-Za-z0-9_]+)\s*>\s*>?\s*"
                         r"(?:\n[^\n]*)*?\n[^\n]*?\b(?P<entry>Boys[A-Za-z0-9_]+)\s*\(")

# A row of the lane contract table. The fields read here are the lane, the name the run
# prints it under, the base figure, the two terms beside it and the member the first of
# them is under; the prose after them is not read.
LANE_ROW = re.compile(
    r"\{\s*Precision::(?P<member>k[A-Za-z0-9_]+)\s*,\s*\"(?P<name>[^\"]*)\"\s*,\s*"
    r"(?P<bound>[-+0-9.eE]+)\s*,\s*(?P<additive>[-+0-9.eE]+)\s*,\s*"
    r"(?P<plain>[-+0-9.eE]+)\s*,\s*RegionBExp::(?P<member_exp>k[A-Za-z0-9_]+)")

# A row of the class list: `X(kHost, kFp64, kAllOrders, FitRoute::..., ...)`. The macro's
# arguments are bare enumerators, not qualified ones, and only the first three cells are
# the class; the rest are that class's default combination.
CLASS_ROW = re.compile(r"^\s*X\(\s*(?P<device>k[A-Za-z0-9_]+)\s*,\s*"
                       r"(?P<precision>k[A-Za-z0-9_]+)\s*,\s*"
                       r"(?P<shape>k[A-Za-z0-9_]+)\s*,", re.M)

# The entries the combination block instantiates, in the library's own spelling.
HOST_ENTRY = re.compile(r"boys::(Boys[A-Za-z0-9_]+)\s*<")
DEVICE_ENTRY = re.compile(r"boys::BoysCuda::(AllOrders[A-Za-z0-9_]*|Single[A-Za-z0-9_]*)")

# A device entry whose own region-B member is a template argument, as an arm spells it:
# `boys::BoysCuda::SingleF32<boys::RegionBExp::kFast>` is the fast row of a name the option
# table also carries at the accurate one. The argument is what the arm read; the bare name
# is the entry's default, which the build states as a seam rather than as a value, so the
# spelled member is read here and the bare name is not guessed at.
DEVICE_ENTRY_SPELLED = re.compile(
    r"boys::BoysCuda::(?P<name>[A-Za-z0-9_]+)\s*<\s*(?:[A-Za-z_][A-Za-z0-9_]*::)*"
    r"(?P<member>k[A-Za-z0-9_]+)\s*>")

# The run's own arithmetic, printed under the combination table: the space read off the
# tables a second way. Read so that this check's universe is held to the run's number as
# well as to the tables.
# The run prints this block wrapped over four lines, so the pattern reads across the
# wrapping with `\s*` rather than requiring one line.
RUN_ARITHMETIC = re.compile(r"the space read off the tables a second way:\s*(?P<total>\d+)\s*"
                            r"member\(s\) over (?P<lanes>\d+)\s*lane\(s\),\s*"
                            r"a route axis of (?P<routes>[0-9 ]+?)\s*route\(s\),\s*"
                            r"(?P<schemes>\d+)\s*scheme\(s\),\s*(?P<partitions>\d+)\s*partition\(s\),\s*"
                            r"(?P<axes>\d+)\s*axis\(es\)")

# The state a row carries. Read from the gate so that a state added there is a state this
# check reads rather than one it silently drops.
STATE = re.compile(r"c\.state\s*=\s*\"(?P<state>[^\"]+)\"")

# The run's own statement that every cell of its cross was read at each of the division
# forms - which is what makes one row a measurement of one combination per form rather
# than of one combination. It is read and not assumed: a run that did not say it would be
# counted at one form, and said to be.
FORM_COVERAGE = re.compile(r"every cell of the cross above was read at each of the\s*"
                           r"(?P<count>\d+)\s*form\(s\) BoysDivisionForms\(\) answers")

# The run's own statement that every *host* cell of its cross was read at each of the
# region-B members the library answers - which is what makes one host row a measurement of
# one combination per member rather than of one combination. The scope word is read with
# the count, because the same statement scoped to a subset of the lanes is a different
# fact: the sentence below it takes the device arms' cells out of the table, and a reader
# that dropped the scope would count those at two members where the run reads them at one.
MEMBER_COVERAGE = re.compile(r"the region-B member:\s*every\s*(?P<scope>[a-z]+)\s*cell of the "
                             r"cross above was read at each of the\s*(?P<count>\d+)\s*"
                             r"member\(s\) BoysRegionBExps\(\) answers")

# ...and the sentence that takes the device arms out of it, which has to be there: a run
# that read its host cells at two members and said nothing about the device ones would
# leave those counted at one member by this check's reading rather than by the run's word.
MEMBER_DEVICE_EXCLUSION = re.compile(
    r"the device arms' cells\s*are not in this table:\s*they are read at the member their\s*"
    r"entry names", re.I)

# The scope word such a sentence carries for the host lanes: the run's own word for the
# cells its member table crosses. The word is read and not dropped - a sentence scoped to
# another set of cells is a statement about those cells, and crediting the host lanes at
# every member off it would be this check's reading of the sentence rather than the
# sentence. A word this check cannot tie to the host lanes is reported, so a run that
# scopes the statement differently is a finding and not a silent second reading of it.
MEMBER_SCOPE_HOST = "host"

# The packing-axis member a one-order shape cannot be instantiated at, named the way the
# gate names it. The library spells the axis; this is the member, and it is read from the
# axis table rather than written here.
ORDERS_MEMBER = "kOrders"

# What makes a lane a device lane: its own member's documentation in the Precision enumeration.
DEVICE_LANE_DOC = re.compile(r"\bdevice\b", re.I)

SCALAR = {"kFp64": "double", "kFp32": "float", "kFp16": "boys::F16"}
BUDGET = {"kFp64": "kFloat", "kFp32": "kFloat", "kFp16": "kFp16"}
VALUE = {"kFp64": "1.5", "kFp32": "1.5f", "kFp16": "boys::F16(1.5f)"}
SUFFIX = {"kFp64": "", "kFp32": "F32", "kFp16": "F16"}

VCVARS = ("C:\\Program Files\\Microsoft Visual Studio\\18\\Community\\VC\\Auxiliary\\Build\\"
          "vcvars64.bat")


def git(*args: str) -> tuple[int, str, str]:
    """git in this repository, as (exit status, stdout, stderr)."""
    try:
        proc = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                              encoding="utf-8", errors="replace")
    except OSError as error:
        return 127, "", f"git could not be run: {error}"

    if proc.stdout is None or proc.stderr is None:
        return 127, proc.stdout or "", (proc.stderr or "").strip() or "git produced no stream"

    return proc.returncode, proc.stdout, proc.stderr.strip()


class Library:
    """The library's facts, read from one revision's files and never from the working tree."""

    def __init__(self, revision: str) -> None:
        self.revision = revision
        self.text: dict[str, str] = {}

        for path in LIBRARY:
            status, out, error = git("show", f"{revision}:{path}")

            if status != 0:
                raise LookupError(f"{revision}:{path} could not be read: {error or 'no reason given'}")

            self.text[path] = out

        self.header = self.text["include/boys/boys.hpp"]
        self.accuracy = self.text["include/boys/accuracy.hpp"]
        self.backend = self.text["include/boys/backend.hpp"]
        self.defaults = self.text["include/boys/boys_build_defaults.hpp"]
        self.options = self.text["include/boys/boys_cuda_options.hpp"]
        self.source = self.text["src/boys.cpp"]
        self.device_source = self.text["src/boys_cuda.cpp"]
        self.all_text = "\n".join(self.text.values())

        # The name functions, which are where each axis member's spelling lives.
        self.names = {match.group("member"): match.group("name")
                      for match in NAME_CASE.finditer(self.source)}

        # The route table's own names, which its rows carry as literals.
        self.names.update({match.group("member"): match.group("name")
                           for match in ROUTE_ROW.finditer(self.source)})

        # The entries and the class each answers for.
        self.entry_class: dict[str, tuple[str, str]] = {}
        for match in ENTRY_CLASS.finditer(self.header):
            self.entry_class.setdefault(match.group("entry"),
                                        (match.group("precision"), match.group("shape")))

        # The lane contract table: one row per lane, with the figure the accessor answers
        # and the region-B member the term beside it is under.
        self.lanes: list[dict] = []
        for match in LANE_ROW.finditer(self.source):
            self.lanes.append({
                "member": match.group("member"),
                "name": match.group("name"),
                "bound": float(match.group("bound")),
                "additive": float(match.group("additive")),
                "plain": float(match.group("plain")),
                "additiveMember": match.group("member_exp"),
            })

        # Which lanes are the device lanes. The Precision enumeration documents each of its
        # members, and the member whose own documentation names the device is a lane whose cells
        # are reached through the CUDA surface; a name spelled `-device` would be this check's
        # idea of the library's split rather than the library's. It is what the run's member
        # sentence is scoped by - `every host cell` - so it is read here and not assumed there.
        precision_docs = self.enum_members("Precision", self.header)

        for lane in self.lanes:
            lane["device"] = bool(DEVICE_LANE_DOC.search(precision_docs.get(lane["member"], "")))

        # The classes the default-policy list names.
        self.classes: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()

        for match in CLASS_ROW.finditer(self.defaults):
            triple = (match.group("device"), match.group("precision"), match.group("shape"))

            if triple not in seen:
                seen.add(triple)
                self.classes.append(triple)

        # The member each device entry runs. On a device lane the member is not a policy
        # argument of the call - the launched entries are distinct functions and the member
        # is the entry - so the one statement of it is this table's own field, and a lane's
        # cells are read at it whatever the lane's contract row states beside its base.
        # Read here rather than taken from that row, because the two disagree on
        # fp32-device: the row's term is under the fast member and the launched entry it
        # arms runs the accurate one, so a credit read off the row names a member the arm
        # never read.
        self.device_row_members: dict[str, set[str]] = {}

        for row in device_option_rows(self.device_source):
            entry = DEVICE_OPTION_ENTRY.search(row)
            member = DEVICE_OPTION_MEMBER.search(row)

            if entry is not None and member is not None:
                self.device_row_members.setdefault(entry.group(1), set()).add(member.group(1))

        # The C++ entry each DeviceEntry enumerator is, read from the enumerator's own
        # documentation, so a name a device arm spells is placed on the table rather than
        # matched by transcription here.
        self.device_entry_members: dict[str, set[str]] = {}

        for member, doc in self.enum_members("DeviceEntry", self.options).items():
            members = self.device_row_members.get(member)

            if members is None:
                continue

            for name in DEVICE_ENTRY_NAME.findall(doc):
                self.device_entry_members.setdefault(name, set()).update(members)

        # The Shape enumerators, with the documentation each carries: a shape whose doc
        # names one order is a shape with no ladder, and the orders axis is not an axis on
        # it. Read rather than written, so a shape added to the enumeration is classified
        # by its own doc.
        self.shapes = self.enum_members("Shape", self.header)
        self.one_order = {name for name, doc in self.shapes.items() if ONE_ORDER.search(doc)}
        self.shape_order = list(self.shapes)

        self.precisions = list(self.enum_members("Precision", self.header))
        self.devices = list(self.enum_members("Device", self.header))
        self.routes = list(self.enum_members("FitRoute", self.accuracy))
        self.schemes = list(self.enum_members("EvalScheme", self.accuracy))
        self.axes = list(self.enum_members("PackAxis", self.backend))
        self.partitions = list(self.enum_members("FitGranularity", self.accuracy))
        self.forms = list(self.enum_members("DivisionForm", self.accuracy))
        self.exps = list(self.enum_members("RegionBExp", self.accuracy))

    @staticmethod
    def enum_members(enum: str, text: str) -> dict[str, str]:
        """The members of one enumeration, as {member: documentation}."""
        start = text.find(f"enum class {enum}")

        if start < 0:
            return {}

        end = text.find("};", start)

        if end < 0:
            return {}

        return {match.group(1): (match.group(2) or "")
                for match in ENUM_MEMBER.finditer(text[start:end])}

    def table_members(self, accessor: str, enum: str) -> set[str] | None:
        """The axis members a table's rows name, or None where its rows name none.

        A table's rows are read out of the accessor that publishes it, and only rows whose
        own enumerator is this table's are read - the qualifier matters, because these
        bodies carry other tables' members under their own enum (`std::max({detail::kBDeg,
        ...})` is a degree and not a partition) and a read that took any `{Enum::kName`
        would report them as this table's.
        """
        where = self.source.find(f"{accessor}()")

        if where < 0:
            return None

        tail = self.source[where:]

        # The body ends at the definition's closing brace - the one at the start of a
        # line, which no nested block's brace is - so a member read here is one this
        # table's rows name and not a later table's.
        closing = re.search(r"\n\}", tail)
        row = re.compile(r"\{\s*" + enum + r"::(k[A-Za-z0-9_]+)")
        names = {match.group(1)
                 for match in row.finditer(tail[:closing.start()] if closing else tail)}

        return names or None

    def shape_of_entry(self, entry: str) -> tuple[str, str] | None:
        """The (precision, shape) an entry answers for, read from its own default policy.

        A device entry is named after its host sibling and carries no policy default of
        its own - `BoysCuda::AllOrdersF32NarrowRatHorner` beside the host's
        `BoysAllOrdersF32` - so the family name is what maps it: the device name under the
        library's own prefix is looked up as an entry. The longest declared entry name that
        is a prefix of it wins, so a name that carries no family is reported as
        unclassified rather than guessed at.
        """
        family = self.family_of_entry(entry)

        return self.entry_class[family] if family is not None else None

    def family_of_entry(self, entry: str) -> str | None:
        """The declared entry an entry's name carries, device names included.

        The device surface spells a host entry's name without the library's prefix, so a
        name that does not begin with the library's own is put back before the lookup:
        `AllOrdersF32` is `BoysAllOrdersF32`. The longest declared name that is a prefix
        wins, so a name that carries no family has none.
        """
        if entry in self.entry_class:
            return entry

        candidate = entry if entry.startswith("Boys") else "Boys" + entry
        best: str | None = None
        longest = 0

        for name in self.entry_class:
            if candidate.startswith(name) and len(name) > longest:
                best, longest = name, len(name)

        return best

    def device_members_of(self, name: str) -> set[str] | None:
        """The region-B members the entry a device arm names runs, or None where no row is it.

        None is a read that failed rather than a lane that runs no member: the name is one
        the option table does not carry, so nothing states what its cells are read at.
        """
        return self.device_entry_members.get(name)

    def device_lane_of(self, name: str) -> str | None:
        """The device lane an entry a device arm names belongs to, read from the name's own family.

        The device surface spells a host entry's name without the library's prefix and with
        the lane's precision in it (`AllOrdersF32NarrowRatHorner` beside `BoysAllOrdersF32`),
        so the family's own precision names the lane - the member of it a device lane is.
        The lane is required to be one the library documents as a device lane, so a name
        whose family resolves to a precision that has no device lane is not placed on one.
        """
        family = self.family_of_entry(name)

        if family is None or family not in self.entry_class:
            return None

        lane = next((row for row in self.lanes
                     if row["member"] == self.entry_class[family][0] + "Device" and row["device"]),
                    None)

        return lane["member"] if lane is not None else None

    def accessor_figure(self, lane: dict, form: str, exp: str) -> float:
        """The figure `BoysAccuracyGuaranteed` answers for a lane at a form and a member.

        The composition is the accessor's own (src/boys.cpp): the lane's base plus the
        term the row publishes for the plain reciprocal, plus the term the row publishes
        beside the base where the member named is the one that term is under.
        """
        form_term = lane["plain"] if form == "kPlainReciprocal" else 0.0
        member_term = lane["additive"] if exp == lane["additiveMember"] else 0.0

        return lane["bound"] + form_term + member_term


def cpp_code(text: str) -> str:
    """C++ source with its comments and string literals blanked out, for structural reads.

    Brace matching over a file that carries a prose comment reads the prose: this library's
    carriage rules are argued in comments that sit inside the function bodies, and the
    argument is not the function. Every character of a comment or a literal is replaced by a
    space rather than removed, so the indices of the code around it do not move and a span
    read here is a span of the original.
    """
    out = list(text)
    i = 0
    n = len(text)

    while i < n:
        if text.startswith("//", i):
            while i < n and text[i] != "\n":
                out[i] = " "
                i += 1
        elif text.startswith("/*", i):
            out[i] = out[i + 1] = " "
            i += 2

            while i < n and not text.startswith("*/", i):
                out[i] = " " if text[i] != "\n" else "\n"
                i += 1

            if i < n:
                out[i] = out[i + 1] = " "
                i += 2
        elif text[i] == '"':
            out[i] = " "
            i += 1

            while i < n and text[i] != '"':
                if text[i] == "\\" and i + 1 < n:
                    out[i] = out[i + 1] = " "
                    i += 2
                    continue
                out[i] = " " if text[i] != "\n" else "\n"
                i += 1

            if i < n:
                out[i] = " "
                i += 1
        else:
            i += 1

    return "".join(out)


def block_end(code: str, opening: int) -> int:
    """The index just past the brace-matched block whose `{` is the first at or after `opening`.

    Read over ``cpp_code``'s output. -1 when no block opens at or after that index, or when
    the braces do not balance, which is how a body this check cannot bound is reported rather
    than silently read to the end of the file.
    """
    start = code.find("{", opening)

    if start < 0:
        return -1

    depth = 0

    for i in range(start, len(code)):
        if code[i] == "{":
            depth += 1
        elif code[i] == "}":
            depth -= 1

            if depth == 0:
                return i + 1

    return -1


def condition_at(code: str, start: int) -> str:
    """The parenthesised condition of the `if` whose `(` is the first at or after `start`.

    Parenthesis-matched rather than read to the first `)`, because these conditions are
    several lines long and each of their tests carries a call of its own.
    """
    opening = code.find("(", start)

    if opening < 0:
        return ""

    depth = 0

    for i in range(opening, len(code)):
        if code[i] == "(":
            depth += 1
        elif code[i] == ")":
            depth -= 1

            if depth == 0:
                return code[opening:i + 1]

    return ""


def array_elements(body: str) -> list[str]:
    """The elements of a brace-initialised array, split at the commas that separate them.

    The body is the text BETWEEN the array's own braces, so an element's braces nest inside
    it and a comma inside one of them separates two of that element's fields rather than two
    elements. An array whose rows are wrapped over lines is read the same way as one written
    a row to a line, which is what the device option table is.
    """
    elements: list[str] = []
    depth = 0
    current: list[str] = []

    for char in body:
        if char in "{[(":
            depth += 1
        elif char in "}])":
            depth -= 1

        if char == "," and depth == 0:
            elements.append("".join(current))
            current = []
            continue

        current.append(char)

    elements.append("".join(current))
    return [element for element in elements if element.strip()]


def device_option_rows(source: str) -> list[str]:
    """The rows of the device option table, each as its own text, or none where it is not found."""
    code = cpp_code(source)
    start = code.find(DEVICE_OPTION_TABLE)

    if start < 0:
        return []

    end = block_end(code, start)

    if end < 0:
        return []

    body = code[start:end]
    opening = body.find("{")

    return array_elements(body[opening + 1:-1]) if opening >= 0 else []


def enum_guard(condition: str, tables: dict[str, str]) -> tuple[list[str], int]:
    """What a carriage guard tests, as (the axes it names, how many sizes it tests).

    A guard on the enumerations compares each axis against its table's own size, so the name of
    the table is what says which axis is being tested. The name is not always in the condition:
    the accessor binds the partition table to a local and tests the local's size, so a read that
    only matched names would report that guard as testing three axes of four and call the
    revision's own source broken. The count of size tests is therefore read as well, and a guard
    is judged on the count - a guard that lost an axis has one size test fewer.
    """
    named = []

    for axis, table in tables.items():
        if f"{table}()" in condition and ">=" in condition:
            named.append(axis)

    return named, len(re.findall(r"\.size\(\)", condition))


def carried_universe(library: "Library") -> tuple[dict[str, str], list[str]]:
    """What `BoysAccuracyGuaranteed` answers `available = true` for, read at the revision.

    The universe this check compares the run against is the set of tuples the library says it
    carries, not the cross of its tables: those are the same set only while no carriage rule
    can refuse a tuple built out of the tables. That is read here rather than assumed - the
    rules' own parameters say which axes they can see, and each rule's refusals are placed
    against its guard - so a rule that could refuse a tuple inside the enumerations is a
    finding, because it would make the gap this check prints wider than the library's.

    Returns the read, as fields to print, and whatever did not hold.
    """
    code = cpp_code(library.source)

    # The axes' widths are the enumerations' own. ``BoysFitGranularities()`` builds its rows
    # rather than listing them, so its member count is not read from the table but from the
    # enumeration the table is a table of - and the row count is read from the table beside
    # it, so the two are compared rather than one standing in for the other.
    routes = library.routes
    axes = library.axes
    partitions = library.partitions
    lanes = library.precisions

    tables = {
        "route": "BoysFitRoutes",
        "scheme": "BoysEvalSchemes",
        "axis": "BoysPackAxes",
        "partition": "BoysFitGranularities",
    }

    # The axes a carriage decision can turn on. The accessor's own doc says the figure does
    # not: `form` and `exp` move the value and are not part of what makes a combination
    # carried, and that is what these two names are checked against.
    arithmetic_axes = set(library.forms) | set(library.exps)

    read: dict[str, str] = {}
    findings: list[str] = []
    covered: set[str] = set()

    def guard_coverage(where: str, body: str) -> set[str]:
        """Every enumeration guard in a body, and whether each one tests the whole enumeration.

        An enumeration guard is one that names a table of the enumeration, or that makes one
        size test per axis - the second clause is what reads a guard whose conditions are bound
        to locals rather than written against the tables. Such a guard has to test all four: a
        value outside the enumeration on an axis it does not test passes the guard and the tuple
        reads as carried. A guard that does neither is a guard on something else - the lane table
        is guarded by a size test too - and is not one of these. Reading the union of the guards
        rather than each of them would hide a guard that lost an axis behind another that still
        has it, which is the failure this is here to catch.
        """
        found: set[str] = set()

        for guard in re.finditer(r"if\s*\(", body):
            named, sizes = enum_guard(condition_at(body, guard.start()), tables)
            whole = len(tables)

            if not named and sizes != whole:
                continue

            found |= set(named)

            if sizes != whole:
                findings.append(
                    f"{where} makes {sizes} size test(s) over the enumerations where this "
                    f"revision's axes number {whole}, so a tuple naming an axis it does not test "
                    f"is a value outside the enumeration that reads as carried")

        return found

    def refusal_placement(where: str, body: str) -> None:
        """Where one carriage rule's refusals sit against its enumeration guard."""
        refusals = [m.start() for m in re.finditer(r"return\s*\{\s*false", body)]
        accepts = len(re.findall(r"return\s*\{\s*true", body))

        if accepts == 0:
            findings.append(f"{where} accepts nothing at all, so no tuple it answers for is "
                            f"carried")

        if not refusals:
            read[f"{where}: guard"] = "makes no refusal"
            return

        guard_at = body.find("if (")

        if guard_at < 0:
            findings.append(f"{where} refuses without a guard, so a tuple built out of the tables "
                            f"can be refused by it and the cross of the tables is wider than what "
                            f"the library carries")
            return

        condition = condition_at(body, guard_at)
        block = block_end(body, body.find("(", guard_at))
        inside = sum(1 for refusal in refusals if block > 0 and refusal < block)
        named, sizes = enum_guard(condition, tables)

        read[f"{where}: guard"] = (
            f"{sizes} size test(s) over the enumerations; the names it writes are "
            f"{', '.join(sorted(named)) if named else 'none'}"
            if sizes else "tests no enumeration")

        if inside != len(refusals):
            findings.append(
                f"{where} makes {len(refusals) - inside} refusal(s) its enumeration guard does "
                f"not cover, so a tuple built out of the tables can be refused by it and the "
                f"cross of the tables is wider than what the library carries")

    for rule in CARRIER.finditer(code):
        name = rule.group("name")
        params = [field.strip() for field in rule.group("params").split(",") if field.strip()]
        seen = [field.split()[0] for field in params if " " in field]
        body_end = block_end(code, rule.end())
        body = code[rule.end():body_end if body_end > 0 else len(code)]

        read[f"{name}: takes"] = ", ".join(seen) if seen else "no axis"
        refusal_placement(f"the carriage rule {name}", body)
        covered |= guard_coverage(f"the carriage rule {name}", body)

        if set(seen) & arithmetic_axes:
            findings.append(
                f"the carriage rule {name} takes an axis the accessor's doc says moves the value "
                f"and not the carriage ({', '.join(sorted(set(seen) & arithmetic_axes))}), so "
                f"the run's reading of that axis as always carried is not the rule's own")

    # The double lane's carriage is decided in the accessor itself rather than by a rule of its
    # own, and the accessor refuses by returning the figure it has filled in rather than by
    # returning a carriage - so it is read by its own shape: what it returns before it answers,
    # and what each of those returns is guarded by.
    accessor_at = code.find("AccuracyFigure BoysAccuracyGuaranteed(")

    if accessor_at < 0:
        findings.append("the accessor's own definition could not be found in the revision's "
                        "source, so what it refuses for the double lane was not read")
    else:
        end = block_end(code, accessor_at)
        body = code[accessor_at:end if end > 0 else len(code)]
        answers = [m.start() for m in re.finditer(r"figure\.available\s*=\s*true", body)]

        if len(answers) != 1:
            findings.append(
                f"BoysAccuracyGuaranteed sets `available = true` {len(answers)} time(s) in its "
                f"own body, so which statement states a combination is carried was not read")
        elif not re.search(r"figure\.available\s*=\s*false", body[:answers[0]]):
            early = [m.start() for m in re.finditer(r"return\s+figure\s*;", body[:answers[0]])]
            guards = []

            for refusal in early:
                guard_at = body.rfind("if (", 0, refusal)

                if guard_at < 0:
                    continue

                depth = 0
                i = guard_at + 3

                while i < len(body):
                    if body[i] == "(":
                        depth += 1
                    elif body[i] == ")":
                        depth -= 1

                        if depth == 0:
                            break
                    i += 1

                guards.append(body[guard_at:i + 1])

            lane_guarded = [g for g in guards if "lanes.size()" in g]
            carriage_guarded = [g for g in guards if "carriage.carried" in g]

            read["the accessor: answers"] = (
                f"{len(early)} return(s) before it answers, "
                f"{len(lane_guarded)} guarded on the lane enumeration and "
                f"{len(carriage_guarded)} on the carriage")

            if len(lane_guarded) != 1 or len(carriage_guarded) != 1 or len(early) != 2:
                findings.append(
                    f"the accessor refuses at {len(early)} point(s) before it answers, "
                    f"{len(lane_guarded)} of them guarded on the lane enumeration and "
                    f"{len(carriage_guarded)} on the carriage, so a tuple built out of the "
                    f"tables can be refused by a guard this check has not read")

            # The double lane has no carriage rule of its own: its enumeration guard is written
            # out in the accessor's own body, so the axes that guard tests are read here rather
            # than left to the rules above, which never see the double lane.
            accessor_axes = guard_coverage("the accessor", body)
            accessor_sizes = max((enum_guard(condition_at(body, guard.start()), tables)[1]
                                  for guard in re.finditer(r"if\s*\(", body)), default=0)

            read["the accessor: guard"] = (
                f"{accessor_sizes} size test(s) over the enumerations; the names it writes are "
                f"{', '.join(sorted(accessor_axes)) if accessor_axes else 'none'}"
                if accessor_sizes else "tests no enumeration")
            covered |= accessor_axes

    read["axes a carriage decision can turn on"] = (
        ", ".join(sorted(covered)) if covered else "none")

    if set(covered) != set(tables):
        findings.append(
            f"the carriage rules guard {', '.join(sorted(covered)) or 'no axis'} where this "
            f"check crosses {', '.join(sorted(tables))}, so a tuple naming an unguarded axis is "
            f"a value outside the enumeration that reads as carried")

    # The double lane's carriage adds the partition rows' own coverage, so a row that leaves a
    # route or an axis clear refuses a tuple that is inside every enumeration.
    bit_values: dict[str, str] = {}

    for definition in BIT_DEFINITION.finditer(code):
        bit_values[definition.group("name")] = definition.group("value").strip()

    def expand(value: str) -> str:
        """A bitmask expression with each name in it replaced by what the name is defined as.

        The rows write a mask two ways - one name (`kBothAxes`) and two names joined
        (`kChebBit | kRatBit`) - so expanding the expression as a whole reads only the first,
        and reads the second as a bitmask with no bits in it. Each name is expanded instead,
        repeatedly, so a name defined in terms of another is reached too.
        """
        for _ in range(4):
            replaced = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b",
                              lambda name: bit_values.get(name.group(0), name.group(0)), value)

            if replaced == value:
                break

            value = replaced

        return value

    for field, pattern in (("routes", PARTITION_ROUTES), ("axes", PARTITION_AXES)):
        for row in pattern.finditer(code):
            value = row.group("value").strip()
            members = len(re.findall(r"1u\s*<<", expand(value)))
            wanted = len(routes) if field == "routes" else len(axes)

            read[f"partition row {row.group('row')}: {field}"] = value

            if members != wanted:
                findings.append(
                    f"partition row {row.group('row')} covers {members} of the "
                    f"{wanted} member(s) of the {field[:-1]} axis, so a tuple naming one it "
                    f"leaves clear is inside every enumeration and is refused by the row")

    built = {int(match.group("row")) for match in PARTITION_ROW_INDEX.finditer(code)}

    read["partition rows built"] = ", ".join(str(row) for row in sorted(built)) or "none"

    if len(built) != len(partitions):
        findings.append(
            f"BoysFitGranularities() builds {len(built)} row(s) where the enumeration it is a "
            f"table of names {len(partitions)}, so the partition axis this check crosses is not "
            f"the table the accessor reads")

    read["lanes"] = f"{len(lanes)} of the enumeration's member(s)"

    # The accessor indexes the lane table by the precision, so the two are one list only while
    # they are the same length; a lane added to the enumeration without a row is a precision the
    # accessor refuses, and one added to the table without an enumerator is a row no caller can
    # name.
    lane_rows = library.table_members("BoysLaneContracts", "Precision")

    if lane_rows is not None and len(lane_rows) != len(lanes):
        findings.append(
            f"BoysLaneContracts() names {len(lane_rows)} lane(s) where the Precision enumeration "
            f"names {len(lanes)}, so the precision axis this check crosses is not the table the "
            f"accessor indexes")

    return read, findings


class RecordedRun:
    """The gate's recorded run, as the combination book it carries."""

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self.text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        self.rows: list[dict] = []
        self.entry_rows: list[dict] = []
        self.entry_rows_note: str | None = None
        self.revision: str | None = None
        self.arithmetic: dict | None = None
        self.forms_per_row: int | None = None
        self.members_per_row: int | None = None
        self.member_scope: str | None = None
        self.member_device_exclusion = False
        self.device_measured: int | None = None
        self.device_carried: int | None = None

        device = DEVICE_LANES.search(self.text)

        if device is not None:
            self.device_measured = int(device.group("measured"))
            self.device_carried = int(device.group("carried"))

        match = REVISION.search(self.text)

        if match is not None:
            self.revision = match.group("rev")

        forms = FORM_COVERAGE.search(self.text)

        if forms is not None:
            self.forms_per_row = int(forms.group("count"))

        members = MEMBER_COVERAGE.search(self.text)

        if members is not None:
            self.members_per_row = int(members.group("count"))
            self.member_scope = members.group("scope").lower()

        self.member_device_exclusion = MEMBER_DEVICE_EXCLUSION.search(self.text) is not None

        arithmetic = RUN_ARITHMETIC.search(self.text)

        if arithmetic is not None:
            self.arithmetic = {
                "total": int(arithmetic.group("total")),
                "lanes": int(arithmetic.group("lanes")),
                "routes": [int(v) for v in arithmetic.group("routes").split()],
                "schemes": int(arithmetic.group("schemes")),
                "partitions": int(arithmetic.group("partitions")),
                "axes": int(arithmetic.group("axes")),
            }

        header = ROW_HEADER.search(self.text)

        if header is not None:
            tail = self.text[header.end():]
            opening = DASHES.search(tail)
            body = tail[opening.end():] if opening is not None else ""
            closing = DASHES.search(body)

            if closing is not None:
                body = body[:closing.start()]

            for match in COMBINATION_ROW.finditer(body):
                parts = [part.strip() for part in match.group("axes").split(",")]

                if len(parts) != 5:
                    continue

                self.rows.append({
                    "lane": parts[0],
                    "route": parts[1],
                    "scheme": parts[2],
                    "partition": parts[3],
                    "axis": parts[4],
                    "cells": int(match.group("cells")),
                    "over": int(match.group("over")),
                    "delivered": match.group("delivered"),
                    "bound": match.group("bound"),
                    "state": match.group("state"),
                    "line": match.group(0).strip(),
                })

        # The entry book, which names the entry a caller reaches rather than the lane.
        opening = ENTRY_BOOK_OPEN.search(self.text)

        if opening is None:
            self.entry_rows_note = "the run carries no entry book: no line of it opens one"
            return

        tail = self.text[opening.end():]
        closing = ENTRY_BOOK_CLOSE.search(tail)
        body = tail[:closing.start()] if closing is not None else tail

        for match in ENTRY_ROW.finditer(body):
            tokens = match.group("rest").split()
            state = ENTRY_COLUMNS.split(match.group(0).strip())[-1]
            entry = " ".join(match.group("entry").split())

            if len(tokens) < 4 or not state:
                continue

            cells = tokens[3]

            if entry == "entry" or not (cells.isdigit() or cells == "n/a"):
                continue

            self.entry_rows.append({
                "entry": entry,
                "route": tokens[0],
                "scheme": tokens[1],
                "axis": tokens[2],
                "cells": int(cells) if cells.isdigit() else 0,
                "measured": cells.isdigit(),
                "state": state,
                "line": match.group(0).strip(),
            })


def cross_entries(gate_text: str) -> tuple[list[str], list[str]]:
    """The entries the gate's combination block instantiates, and its device entries.

    The block is the gate's own: it opens with its marker comment and closes at the next
    one. A block that cannot be located is reported as such rather than read as an empty
    one - a block with no entries in it would otherwise read as a gate that measures
    nothing, which is a different finding.
    """
    opening = BLOCK_OPEN.search(gate_text)

    if opening is None:
        return [], []

    closing = BLOCK_CLOSE.search(gate_text, opening.end())

    if closing is None:
        # No book "counted apart" follows: fall back to the next section marker, and the
        # caller reports the shorter read rather than silently reading a longer one.
        fallback = BLOCK_NEXT.search(gate_text, opening.end())
        closing = fallback

    limit = closing.start() if closing is not None else len(gate_text)
    block = gate_text[opening.start():limit]
    host = sorted({match.group(1) for match in HOST_ENTRY.finditer(block)})

    # The device arms reach their entries through the mapper functions the block calls,
    # whose bodies are outside it: read the names those functions return.
    device: list[str] = []
    for mapper in sorted({m.group(1) for m in re.finditer(r"\b(GateDeviceEntry[A-Za-z0-9_]*)", block)}):
        where = gate_text.find(f"constexpr auto {mapper}(")

        if where < 0:
            continue

        tail = gate_text[where:]
        stop = tail.find("\n}\n")

        if stop < 0:
            stop = min(len(tail), 40000)

        device.extend(match.group(1) for match in DEVICE_ENTRY.finditer(tail[:stop]))

    return host, sorted(set(device))


def state_strings(gate_text: str) -> list[str]:
    """The states the gate sets on a combination row, read from the gate."""
    return sorted({match.group("state") for match in STATE.finditer(gate_text)})


def entry_book_declared(gate_text: str, labels: set[str],
                        declared: set[str]) -> dict[str, str]:
    """The entries the gate's entry book crosses, as {the name it prints: the entry}.

    The book's own comment declares them - `plane entry   BoysAllN(nmax, x, out, count,
    workspace)` - so the entries it crosses are read from the gate's statement of them and
    not from a list here. Only the labels the run actually printed are read, so the prose
    the same comment carries cannot enter the mapping. A row the book prints whose label
    this does not carry is a row this check cannot attribute to an entry, and the caller
    says so.
    """
    opening = ENTRY_BOOK_MARKER.search(gate_text)

    if opening is None:
        return {}

    tail = gate_text[opening.end():]
    following = BLOCK_NEXT.search(tail)
    block = tail[:following.start()] if following is not None else tail
    found: dict[str, str] = {}
    matches = list(ENTRY_BOOK_ENTRY.finditer(block))
    previous: str | None = None

    for position, match in enumerate(matches):
        label = " ".join(match.group("label").split())

        if label not in labels:
            continue

        # The label's own text runs to the next label, and it is that text - not the one
        # line the label sits on - that names the entry.
        end = matches[position + 1].start() if position + 1 < len(matches) else len(block)
        text = match.group("text") + " " + block[match.end():end]
        entry = next((name for name in re.findall(r"Boys[A-Za-z0-9_]+", text)
                      if name in declared), None)

        # An overload of the entry above it is declared by that relation rather than by a
        # second name - `the same entry under its BoysSortedArgs overload` - so the entry
        # it reaches is the one above, read from the gate's own words for it.
        if entry is None and previous is not None:
            entry = previous

        if entry is not None:
            found.setdefault(label, entry)
            previous = entry

    return found


def materialize(revision: str, scratch: pathlib.Path) -> pathlib.Path | None:
    """The revision's include tree, written into a scratch directory.

    The control compiles a translation unit that includes the library's headers, and a header in
    the working tree may have a writer in it. A compile against that tree is a claim about
    whatever the writer had reached, so the tree is rebuilt from the revision the run names and
    the compile reads that - the same rule the library facts are read under.
    """
    status, listing, _ = git("ls-tree", "-r", "--name-only", revision, "include")

    if status != 0:
        return None

    root = scratch / "revision"

    for path in listing.splitlines():
        path = path.strip()

        if not path:
            continue

        status, text, _ = git("show", f"{revision}:{path}")

        if status != 0:
            return None

        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", errors="replace")

    include = root / "include"
    return include if include.is_dir() else None


def compiles(source: str, scratch: pathlib.Path, include: pathlib.Path) -> bool:
    """Whether one scratch translation unit compiles. The negative control of this check."""
    unit = scratch / "unit.cpp"
    unit.write_text(source, encoding="utf-8")
    script = scratch / "run.bat"
    script.write_text(
        "@echo off\n"
        f'call "{VCVARS}" >nul 2>&1\n'
        f'cl /nologo /std:c++latest /EHsc /Zs /I "{include}" "{unit}" >nul 2>&1\n'
        "exit /b %errorlevel%\n",
        encoding="utf-8",
    )

    try:
        completed = subprocess.run(["cmd.exe", "/c", str(script)], capture_output=True, timeout=600)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

    return completed.returncode == 0


def policy(precision: str, pack: str) -> str:
    return (f"boys::EvalPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kHorner, "
            f"boys::BoysBudget::{BUDGET[precision]}, boys::PackAxis::{pack}, "
            f"boys::FitGranularity::kNarrow, boys::DivisionForm::kRefinedReciprocal, "
            f"boys::RegionBExp::kFast>")


def unit_for(precision: str, shape: str, pack: str, entries: dict[str, str]) -> str:
    entry = entries[shape]
    suffix = SUFFIX[precision]
    scalar = SCALAR[precision]
    # The calls are made as statements and cast to void: the entries of these two shapes
    # do not return the same thing - the single entry answers a value and the fixed-order
    # entry writes its values out - and what is being instantiated is the entry at the
    # policy, not the use of a result.
    body = {
        "kSingle": f"(void)boys::{entry}{suffix}<{policy(precision, pack)}>(4, x);",
        "kFixedN": f"(void)boys::{entry}{suffix}<{policy(precision, pack)}>(4, xs, out, 2, 1);",
    }[shape]

    return ("#include <boys/boys.hpp>\n"
            "void Instantiate() {\n"
            f"    [[maybe_unused]] {scalar} x = {VALUE[precision]};\n"
            f"    [[maybe_unused]] {scalar} xs[8] = {{}};\n"
            f"    [[maybe_unused]] {scalar} out[64] = {{}};\n"
            f"    {body}\n"
            "}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=str(RUN), help="the gate's recorded run")
    parser.add_argument("--gate", default=str(GATE), help="the gate itself")
    parser.add_argument("--revision", default=None,
                        help="the revision to read the library at (default: the one the run names)")
    parser.add_argument("--full", action="store_true",
                        help="print every class-and-combination line rather than the grouped gaps")
    parser.add_argument("--compile", action="store_true",
                        help="run the compile control: it proves the one-order exclusion rather than "
                             "asserting it, and costs minutes per instantiation on this machine")
    parser.add_argument("--check", action="store_true",
                        help="accepted for the invocation the workflow uses uniformly; this tool has "
                             "no other mode, so a combination the run does not measure exits "
                             "non-zero with or without it")
    args = parser.parse_args()

    findings: list[str] = []
    run = RecordedRun(pathlib.Path(args.run))

    print("the accuracy gate's recorded run, and the combinations it measures")
    print(f"  run  : {pathlib.Path(args.run)}")
    print(f"  gate : {pathlib.Path(args.gate)}")

    if not run.text:
        print(f"\nSTALE: there is no recorded run to read: {args.run} is not a file. This check's "
              "whole subject is what that file records, so an empty read is not a pass")
        return 1

    revision = args.revision or run.revision

    if revision is None:
        print(f"\nSTALE: {args.run} names no revision: no line of it reads `accuracy gate, "
              "revision <rev>`, which is what dates the run and what says which library's tables "
              "its rows were produced from")
        return 1

    status, resolved, error = git("rev-parse", "--verify", "--quiet", f"{revision}^{{commit}}")

    if status != 0 or not resolved.strip():
        print(f"\nSTALE: the run names revision {revision} and this repository cannot resolve it"
              + (f" ({error})" if error else ""))
        return 1

    print(f"  library read at: {revision} (the revision the run names)")

    try:
        library = Library(revision)
    except LookupError as problem:
        print(f"\nSTALE: {problem}")
        return 1

    if not run.rows:
        print(f"\nSTALE: {args.run} carries no combination rows: no table under a header reading "
              "`combination ... cells outside delivered bound state` was found, so this check has "
              "nothing to hold the library's space to. A read that came back empty is not a pass")
        return 1

    # A record that does not state how much of the device surface its build
    # reached cannot be told from one that reached all of it, and the gap this
    # check then reports would be read as the library's rather than the record's.
    if run.device_measured is None:
        print(f"\nSTALE: {args.run} states no device-lane count: no line of it reads `device "
              f"lanes : N of M measured on a card`, which is what the gate prints for what its "
              f"build carried and its card ran. Re-make the run and commit the new record")
        return 1

    # What the library says it carries, read from the accessor's own rules rather than taken
    # from the cross of the tables. The two are the same set only while no rule can refuse a
    # tuple built out of the tables, and that is what this read is for.
    carriage_read, carriage_findings = carried_universe(library)
    findings.extend(carriage_findings)

    print("\nthe universe the gap is measured against: what the library says it carries")
    print(f"  a tuple is carried when BoysAccuracyGuaranteed answers `available = true` for it,")
    print(f"  and the accessor answers from a carriage rule. Read at {revision} (src/boys.cpp):")

    for field, value in carriage_read.items():
        print(f"    {field:<44} {value}")

    # --------------------------------------------------------------------- the gate
    gate_text = pathlib.Path(args.gate).read_text(encoding="utf-8", errors="replace") \
        if pathlib.Path(args.gate).is_file() else ""
    states = state_strings(gate_text) if gate_text else []
    host_entries, device_entries = cross_entries(gate_text) if gate_text else ([], [])

    if not gate_text:
        findings.append(f"the gate could not be read at {args.gate}, so this check cannot say "
                        "which entry the recorded run's rows were measured through")
    elif not host_entries and not device_entries:
        findings.append("the gate's combination block names no entry: the block's marker comment "
                        "was not found, or the block between it and the next section marker "
                        "instantiates nothing. A block with no entries would leave every class "
                        "unattributed, which is a read that failed rather than a gate that "
                        "measures nothing")
    elif not states:
        findings.append("the gate's combination rows set no state this check could read, so a row's "
                        "state cannot be told from prose")

    # Which class each entry the block instantiates answers for.
    cross_shapes: dict[str, set[str]] = {}
    unclassified: list[str] = []

    for name, where in [(entry, "host") for entry in host_entries] + \
                       [(entry, "device") for entry in device_entries]:
        triple = library.shape_of_entry(name)

        if triple is None:
            unclassified.append(name)
            continue

        cross_shapes.setdefault(where, set()).add(triple[1])

    if unclassified:
        findings.append("the combination block instantiates "
                        + ", ".join(sorted(unclassified))
                        + ", and no declared entry of this revision is a prefix of "
                          + ("that name" if len(unclassified) == 1 else "those names")
                        + ": the class those entries answer for cannot be read from the library, so "
                          "the classes this run measures cannot be stated")

    # The region-B member a device lane's cells were read at: the members the entries the
    # block arms that lane with run. The member is the entry's own on a device lane - the
    # launched entries are distinct functions and no policy argument moves it - so the one
    # statement of it is the device option table, and the lane's contract row states
    # something else: the member its term beside the base belongs to. The two are one member
    # only while the entry the block arms runs the member that term is under, and on
    # fp32-device they are not - the arms name the launched all-orders entry, which runs the
    # accurate member, and the row's 8e-8 term is under the fast one.
    #
    # The member an arm's own spelling states, where it states one: a name the table carries
    # twice is one class at two members, and the arm that reaches the second spells it.
    spelled_members: dict[str, set[str]] = {}

    for match in DEVICE_ENTRY_SPELLED.finditer(gate_text):
        spelled_members.setdefault(match.group("name"), set()).add(match.group("member"))

    device_members: dict[str, list[str]] = {}
    unplaced_devices: list[str] = []
    ambiguous_devices: list[str] = []
    misspelled_devices: list[str] = []

    for name in device_entries:
        names = library.device_members_of(name)
        lane = library.device_lane_of(name)

        if names is None or lane is None:
            unplaced_devices.append(name)
            continue

        spelled = spelled_members.get(name, set())
        members = spelled & names if spelled else names

        if spelled and not members:
            misspelled_devices.append(f"{name} at {', '.join(sorted(spelled))}")
            continue

        # A name the table carries at more than one member is an entry that IS its member -
        # the f32 single entry is one name at two rows - and an arm that names it reaches one
        # of them, which the arm's own spelling states and its bare name does not: the bare
        # name is the entry's default, and the build states that default as a seam rather
        # than as a value. Crediting both would be a credit for a member a particular
        # arm may never have run, which is the credit this whole read exists to close.
        if len(members) > 1:
            ambiguous_devices.append(f"{name} ({', '.join(sorted(names))})")
            continue

        for member in sorted(members):
            if member not in device_members.setdefault(lane, []):
                device_members[lane].append(member)

    if unplaced_devices:
        findings.append(
            "the gate's combination block arms the device lanes with "
            + ", ".join(sorted(unplaced_devices))
            + ", and this revision's device option table carries no row for "
            + ("that name" if len(unplaced_devices) == 1 else "those names")
            + " under a precision this revision documents as a device lane: the member those "
              "arms read is not stated by the library, so a lane with no placed arm of its own "
              "is counted at the member its contract row publishes its term under, which is "
              "where that term belongs and not where an entry is")

    if misspelled_devices:
        findings.append(
            "the gate's combination block arms the device lanes with "
            + ", ".join(sorted(misspelled_devices))
            + ", and the device option table carries "
            + ("that name" if len(misspelled_devices) == 1 else "those names")
            + " at no such member: the member an arm like that reads is not one the library "
              "states for the entry, so its cells are not credited at a member here")

    if ambiguous_devices:
        findings.append(
            "the gate's combination block arms the device lanes with "
            + ", ".join(sorted(ambiguous_devices))
            + ", and the device option table carries "
            + ("that name" if len(ambiguous_devices) == 1 else "those names")
            + " at more than one member: which of them an arm read is the entry's default and "
              "the arm's spelling of it, and a bare name states neither, so those arms are not "
              "credited at either member here rather than at both")

    for lane in library.lanes:
        if lane["device"]:
            device_members.setdefault(lane["member"], [lane["additiveMember"]])

    print(f"\nthe region-B member each device lane's cells are read at, read from the entries the")
    print(f"  block arms that lane with and from the device option table at {revision} "
          f"(src/boys_cuda.cpp):")

    for lane in library.lanes:
        if not lane["device"]:
            continue

        armed = [name for name in device_entries if library.device_members_of(name) is not None
                 and library.device_lane_of(name) == lane["member"]]
        credited = device_members.get(lane["member"], [])
        row_member = library.names.get(lane["additiveMember"], lane["additiveMember"])

        # The figure the run's own rows for this lane record, held to the figures the
        # accessor answers for the member credited and for the other one. A row records the
        # figure it was judged by, and a lane judged at the other member's figure is a lane
        # whose credited member is not the one its bound was measured against - a fact about
        # the run and not about the credit, so it is printed beside the credit rather than
        # counted as a gap.
        figures = {float(row["bound"]) for row in run.rows if row["lane"] == lane["name"]}
        at = {member: {library.accessor_figure(lane, form, member) for form in library.forms}
              for member in library.exps}
        matched = [member for member in library.exps if figures & at[member]]
        judged = ""

        if matched and all(member not in credited for member in matched):
            judged = ("; the figure its rows record is the other member's, which is not the "
                      "member they read")

        print(f"    {lane['name']:<13} {len(armed):>3} entry(ies) -> "
              + ", ".join(library.names.get(member, member) for member in credited)
              + f"   (the lane's row publishes its term under {row_member}{judged})")

    # ------------------------------------------------------------------- the universe
    axis_sizes = {
        "lane": len(library.lanes),
        "route": len(library.routes),
        "scheme": len(library.schemes),
        "axis": len(library.axes),
        "granularity": len(library.partitions),
        "form": len(library.forms),
        "exp": len(library.exps),
    }

    print(f"\nthe axes, read from the library's own tables at {revision}"
          f" (the name a run prints is in brackets)")
    print(f"  lanes       {axis_sizes['lane']}  "
          + " ".join(f"{row['member']} [{row['name']}]" for row in library.lanes))

    for label, axis in (("routes", "route"), ("schemes", "scheme"), ("axes", "axis"),
                        ("partitions", "granularity"), ("forms", "form"), ("exps", "exp")):
        members = {"route": library.routes, "scheme": library.schemes, "axis": library.axes,
                   "granularity": library.partitions, "form": library.forms,
                   "exp": library.exps}[axis]
        shown = [f"{member} [{library.names.get(member, member)}]" for member in members]
        print(f"  {label:<11} {axis_sizes[axis]}  " + " ".join(shown))

    # The tables' own rows, held to the enumerations they index: the axes are the
    # library's tables, so what an axis has and what its table carries have to be the
    # same members. A table whose rows are built by a loop names none of them here, and
    # that is said rather than passed over - an axis whose table cannot be read is an
    # axis whose size rests on the enumeration alone.
    enumerated = {"route": library.routes, "scheme": library.schemes, "axis": library.axes,
                  "granularity": library.partitions, "form": library.forms, "exp": library.exps,
                  "lane": [row["member"] for row in library.lanes]}

    for accessor, (enum, axis) in TABLE_ENUM.items():
        members = library.table_members(accessor, enum)

        if members is None:
            print(f"  note: {accessor}() lists no row per member this check can read (its rows "
                  f"are built rather than written out), so the {axis} axis's size rests on the "
                  f"enumeration and on the run's own arithmetic line below")
            continue

        missing = [member for member in enumerated[axis] if member not in members]
        extra = sorted(member for member in members if member not in enumerated[axis])

        if missing or extra:
            findings.append(f"{accessor}() names {', '.join(extra) if extra else 'no'} member(s) "
                            f"the {enum} enumeration does not have and leaves "
                            f"{', '.join(missing) if missing else 'none'} of it unnamed: a table "
                            "and the enumeration it indexes state different spaces, and every "
                            "count below is one of the two")

    accessor_total = 1
    for size in axis_sizes.values():
        accessor_total *= size

    # ------------------------------------------------------- what the run accounts for
    lane_by_name = {row["name"]: row for row in library.lanes}
    axes_by_name = {
        "route": {library.names.get(member, member): member for member in library.routes},
        "scheme": {library.names.get(member, member): member for member in library.schemes},
        "partition": {library.names.get(member, member): member for member in library.partitions},
        "axis": {library.names.get(member, member): member for member in library.axes},
    }

    # Whether the run states that its host cells were read at every region-B member: its
    # own sentence, with its own scope word and its own count, held to the members the
    # library answers. Both halves are needed and either alone is not the statement.
    host_members_read = (run.members_per_row == len(library.exps)
                         and run.member_scope == MEMBER_SCOPE_HOST)

    def members_credited(lane: dict) -> list[str]:
        """The region-B members one lane's rows were read at.

        A row is read at every region-B member its cells were read at, and which those are is
        a fact about the lane rather than about the row. On a host lane it is every member
        the run's own sentence covers the lane at, and the sentence is scoped: a run whose
        member sentence is about other cells than the host lanes' covers the host lanes at
        the one member their row's figure is composed under, which is where the lane's own
        member table starts and not what the sentence says. On a device lane it is the
        members the entries the block arms that lane with run, read from the device option
        table. It is not the lane's `additiveMember`: the term a row publishes beside its
        base belongs to the member that term is under, and the entrance a device lane's cells
        go through is a function that runs one member of its own. The two are the same member
        where the arms name the entry the term is under and different members where they do
        not, so the credit is read where the entry is.
        """
        if lane["device"]:
            return list(device_members.get(lane["member"], [lane["additiveMember"]]))

        return list(library.exps) if host_members_read else [lane["additiveMember"]]

    lane_credited = {lane["member"]: members_credited(lane) for lane in library.lanes}

    measured: dict[tuple[str, str, str, str, str, str, str], dict] = {}
    unrunnable: list[dict] = []
    unknown: list[str] = []

    for row in run.rows:
        lane = lane_by_name.get(row["lane"])

        if lane is None:
            unknown.append(row["line"])
            continue

        members = {}
        ok = True

        for axis, spelling in (("route", row["route"]), ("scheme", row["scheme"]),
                               ("partition", row["partition"]), ("axis", row["axis"])):
            member = axes_by_name[axis].get(spelling)

            if member is None:
                ok = False
                break

            members[axis] = member

        if not ok:
            unknown.append(row["line"])
            continue

        if row["cells"] == 0:
            unrunnable.append(row)
            continue

        # A row is read at every division form the axis carries - the run's own words, read
        # above and not assumed - and at every region-B member its lane's cells were read at,
        # which `members_credited` states once for the count below and for the gap after it.
        for form in (library.forms if run.forms_per_row is not None
                     else library.forms[:1]):
            for exp in lane_credited[lane["member"]]:
                key = (lane["member"], members["route"], members["scheme"], members["axis"],
                       members["partition"], form, exp)
                measured[key] = row

    if unknown:
        findings.append(f"{len(unknown)} row(s) of the recorded run name an axis member no table of "
                        f"revision {revision} carries (first: {unknown[0]}). A row this check cannot "
                        "place is a row whose combination it cannot count as measured")

    if unrunnable:
        findings.append(f"{len(unrunnable)} row(s) of the recorded run exercised no cell at all "
                        f"(first: {unrunnable[0]['line']}). A row that measured nothing covers "
                        "nothing, whatever state it carries")

    if run.forms_per_row is None:
        findings.append(f"{args.run} does not state that each of its rows was read at every "
                        f"division form, so each row is counted as a measurement of one form "
                        f"here. A row's cells column counts points over the forms, but the "
                        f"statement that every form was read is the run's and this one does not "
                        f"carry it")
    elif run.forms_per_row != len(library.forms):
        findings.append(f"{args.run} states each of its cells was read at {run.forms_per_row} "
                        f"division form(s) where BoysDivisionForms() answers "
                        f"{len(library.forms)}: the forms this check counts a row at are not the "
                        f"forms the run says it read")

    if run.members_per_row is None:
        findings.append(f"{args.run} does not state that its host cells were read at every "
                        f"region-B member, so each row is counted as a measurement of the one "
                        f"member its lane's figure is composed under. The statement that both "
                        f"members were read is the run's, and this one does not carry it")
    elif run.members_per_row != len(library.exps):
        findings.append(f"{args.run} states each of its host cells was read at "
                        f"{run.members_per_row} region-B member(s) where BoysRegionBExps() answers "
                        f"{len(library.exps)}: the members this check counts a row at are not the "
                        f"members the run says it read")

    if run.members_per_row is not None and run.member_scope != MEMBER_SCOPE_HOST:
        findings.append(f"{args.run} scopes its region-B member sentence to `"
                        f"{run.member_scope or 'no word this check could read'}` cells, and this "
                        f"check reads that sentence for the host lanes - the ones the run's own "
                        f"member table crosses. A sentence scoped to another set of cells is a "
                        f"statement about those cells, so the host lanes are counted at one "
                        f"member here rather than credited at every member off it")

    if run.members_per_row is not None and not run.member_device_exclusion:
        findings.append(f"{args.run} states its host cells were read at every region-B member and "
                        f"says nothing about the device lanes' cells, so this check has no word of "
                        f"the run's to count those at one member: they are counted at one because "
                        f"nothing states the other")

    # ----------------------------------------------------- the accessor's own statement
    universe = [(lane["member"], route, scheme, axis, partition, form, exp)
                for lane in library.lanes
                for route in library.routes
                for scheme in library.schemes
                for axis in library.axes
                for partition in library.partitions
                for form in library.forms
                for exp in library.exps]

    covered = [key for key in universe if key in measured]
    uncovered = [key for key in universe if key not in measured]

    # The exp axis: an axis the run never crosses. At one form the two members answer one
    # figure on a lane whose row publishes no term beside the base, and two figures where
    # it does. The comparison is made at a fixed form, because the forms answer different
    # figures from each other on their own account - which is a fact about the form axis
    # and not about this one.
    distinguishing: list[str] = []
    same_figure: list[str] = []

    for lane in library.lanes:
        at = {exp: library.accessor_figure(lane, "kExactDivision", exp) for exp in library.exps}

        if len(set(at.values())) > 1:
            distinguishing.append(f"{lane['name']} ({', '.join(f'{member[1:]}={value:.6g}' for member, value in at.items())})")
        else:
            same_figure.append(lane["name"])

    print(f"\nthe accessor's space: BoysAccuracyGuaranteed over {len(axis_sizes)} axes")
    print(f"  {' x '.join(str(size) for size in axis_sizes.values())} = {accessor_total} "
          f"combination(s), {accessor_total // axis_sizes['lane']} per lane")
    host_lanes = [lane for lane in library.lanes if not lane["device"]]
    device_lanes = [lane for lane in library.lanes if lane["device"]]
    members_host = len(library.exps) if (run.members_per_row == len(library.exps)) else 1

    print(f"  measured by the recorded run : {len(covered)}"
          f"   ({len(run.rows)} row(s) x {run.forms_per_row or 1} form(s) x "
          f"{members_host} region-B member(s) on the {len(host_lanes)} host lane(s), "
          f"1 on the {len(device_lanes)} device lane(s), whose arms are read at the member "
          f"their entry names)")
    where = ("the same rows at the other region-B member" if run.members_per_row is None else
             "this run's own member sentence takes the device lanes' cells out of its table, and "
             "those are counted at the one member their entry names")
    print(f"  not measured                 : {len(uncovered)} of {accessor_total}"
          f"   ({where})")

    if run.arithmetic is not None:
        lanes = run.arithmetic["lanes"]
        product = sum(run.arithmetic["routes"]) * run.arithmetic["schemes"] * \
            run.arithmetic["partitions"] * run.arithmetic["axes"]

        print(f"  the run's own arithmetic     : {run.arithmetic['total']} member(s) over {lanes} "
              f"lane(s), a route axis of {' '.join(str(v) for v in run.arithmetic['routes'])} "
              f"route(s), {run.arithmetic['schemes']} scheme(s), {run.arithmetic['partitions']} "
              f"partition(s), {run.arithmetic['axes']} axis(es)")
        print(f"  read off the tables here     : {lanes} lane(s) x "
              f"{run.arithmetic['routes']} route(s) x {run.arithmetic['schemes']} scheme(s) x "
              f"{run.arithmetic['partitions']} partition(s) x {run.arithmetic['axes']} axis(es) "
              f"= {product}")

        if product != run.arithmetic["total"]:
            findings.append(f"the run states {run.arithmetic['total']} combination(s) over its five "
                            f"axes and its own axis sizes multiply to {product}: the run's arithmetic "
                            "does not close against its own line, so the space it measured is not the "
                            "space its tables report")

    # ------------------------------------------------------------------- per class
    classes: list[dict] = []

    for device, precision, shape in library.classes:
        lane = next((row for row in library.lanes if row["member"] == precision), None)

        if lane is None:
            findings.append(f"the class list names Precision::{precision} and BoysLaneContracts has "
                            "no row for it, so no figure of that class's lane can be read")
            continue

        axes = [axis for axis in library.axes if not (shape in library.one_order and
                                                      axis == ORDERS_MEMBER)]

        space = [(lane["member"], route, scheme, axis, partition, form, exp)
                 for route in library.routes
                 for scheme in library.schemes
                 for axis in axes
                 for partition in library.partitions
                 for form in library.forms
                 for exp in library.exps]

        # Which entries a class's cells could have been measured through is the question its
        # own machine answers: a device class's entry is a device entry, and the host
        # entries' shapes are entries the device lane's rows never go through. The machine is
        # the class row's own first cell, and the shapes are the ones the block arms THAT
        # machine with - so a device class whose shape only a host entry answers for is a
        # class the run never asked rather than one it measured.
        machine = device[1:].lower()

        classes.append({
            "device": device,
            "machine": machine,
            "precision": precision,
            "shape": shape,
            "lane": lane,
            "space": space,
            "shapes": sorted(cross_shapes.get(machine, set())),
            "shape_measured": shape in cross_shapes.get(machine, set()),
        })

    # A machine the class list names and the block arms with no entry at all is a machine
    # whose classes this check cannot attribute: every one of them would be printed as
    # measured through another machine's entry, and which entry produced the cells counted
    # beside it would be a fact nothing read.
    unarmed = sorted({entry["machine"] for entry in classes
                      if not cross_shapes.get(entry["machine"])})

    if unarmed:
        findings.append("the class list names classes of "
                        + ", ".join(unarmed)
                        + ", and the gate's combination block names no entry of "
                        + ("that machine" if len(unarmed) == 1 else "those machines")
                        + ": the entry the cells counted for those classes were measured through "
                          "cannot be read from the block, so those classes are not attributed")

    class_total = sum(len(entry["space"]) for entry in classes)
    class_covered = sum(1 for entry in classes for key in entry["space"] if key in measured)

    print(f"\nper class: the combinations the class carries, how many of them the recorded run")
    print(f"measured a cell at, and the entry it measured those cells through")
    print(f"  {'class':<32} {'combos':>7} {'measured':>9} {'not measured':>13}  "
          f"{'the run measured them through':<40} figures")
    print(f"  {'-' * 32} {'-' * 7} {'-' * 9} {'-' * 13}  {'-' * 40} {'-' * 22}")

    own = 0

    for entry in classes:
        keys = [key for key in entry["space"] if key in measured]
        figures = sorted({float(measured[key]["bound"]) for key in keys})
        shown = (", ".join(f"{value:.6g}" for value in figures[:3])
                 + (f", +{len(figures) - 3} more" if len(figures) > 3 else "")) if figures else "-"

        if entry["shape_measured"]:
            through = "its own shape's entry"
            own += 1 if keys else 0
        elif entry["shapes"]:
            through = ("another class's entry: one of "
                       + ", ".join(f"Shape::{shape}" for shape in entry["shapes"]))
        else:
            through = "no entry of this machine's shapes"

        print(f"  {entry['device'][1:] + ' ' + entry['lane']['name'] + ' ' + entry['shape'][1:]:<32} "
              f"{len(entry['space']):>7} {len(keys):>9} {len(entry['space']) - len(keys):>13}  "
              f"{through:<40} {shown}")

    print(f"  {'-' * 32} {'-' * 7} {'-' * 9} {'-' * 13}")
    print(f"  {'total':<32} {class_total:>7} {class_covered:>9} {class_total - class_covered:>13}")
    print(f"  {own} of {len(classes)} class(es) have a combination measured through an entry of "
          f"their own shape; the cells counted for the other "
          f"{len(classes) - own} belong to rows measured through another class's entry and the")
    print(f"  geometry of this run does not say the two answer the same - it says the run never "
          f"asked this class's entry at all")

    # ------------------------------------------------------------- the covered ones
    print(f"\nthe covered ones, named")
    print(f"  the combination book: {len(run.rows)} row(s), each a (lane, route, scheme, "
          f"partition, packing axis)")
    print(f"  the entry each row is measured through, read from the block that instantiates it:")

    # The device surface is one name per (route, partition, packing axis) per precision, so
    # the names are grouped by the family each one belongs to rather than listed: the
    # family is the library entry the names answer for, and that is the fact this check
    # needs.
    for machine, names in (("host", host_entries), ("device", device_entries)):
        if not names:
            print(f"    {machine:<7}  0  none")
            continue

        families: dict[str, list[str]] = {}

        for name in names:
            families.setdefault(library.family_of_entry(name) or "(no declared family)",
                                []).append(name)

        print(f"    {machine:<7} {len(names):>2}  " + "; ".join(
            f"{family} ({len(members)} name(s))" for family, members in sorted(families.items())))

    entry_shapes: dict[str, set[str]] = {}
    unplaced: list[str] = []

    for name, machine in [(entry, "host") for entry in host_entries] + \
                         [(entry, "device") for entry in device_entries]:
        triple = library.shape_of_entry(name)

        if triple is None:
            unplaced.append(name)
            continue

        entry_shapes.setdefault(machine, set()).add(triple[1])

    if entry_shapes:
        for machine, shapes in sorted(entry_shapes.items()):
            print(f"    -> the {machine} entries answer for "
                  + ", ".join(f"Shape::{shape}" for shape in sorted(shapes)))

        print(f"  so the class a row of this book measures is the one of its lane whose shape the "
              f"entries of its own machine answer for, and a class of that lane whose shape none "
              f"of them answers for is measured by no row of this book")

    print(f"  the entry book: the run's other book, and the only one keyed by the entry a caller")
    print(f"  names rather than by the lane")
    declared_entries = (entry_book_declared(gate_text,
                                           {row["entry"] for row in run.entry_rows},
                                           set(library.entry_class)) if gate_text else {})

    if not run.entry_rows:
        print(f"    {run.entry_rows_note or 'the run carries no entry-book row this check could read'}")
    else:
        print(f"    {'entry':<22} {'Shape':<18} {'route':<18} {'scheme':<16} {'axis':<11} state")
        print(f"    {'-' * 22} {'-' * 18} {'-' * 18} {'-' * 16} {'-' * 11} {'-' * 34}")

        for row in run.entry_rows:
            entry = declared_entries.get(row["entry"])
            triple = library.shape_of_entry(entry) if entry else None
            shape = f"Shape::{triple[1]}" if triple else "unread"

            print(f"    {row['entry']:<22} {shape:<18} {row['route']:<18} {row['scheme']:<16} "
                  f"{row['axis']:<11} {row['state']}")

        unmatched = sorted({row["entry"] for row in run.entry_rows
                            if row["entry"] not in declared_entries})

        if unmatched:
            findings.append("the entry book's rows name "
                            + ", ".join(unmatched)
                            + ", and the gate's comment declares no entry under "
                            + ("that name" if len(unmatched) == 1 else "those names")
                            + ": the shape those rows measure cannot be read from the gate")

        measured_entries = [row for row in run.entry_rows if row["measured"]]
        entry_shapes_book = sorted({library.shape_of_entry(declared_entries[row["entry"]])[1]
                                    for row in measured_entries
                                    if declared_entries.get(row["entry"]) and
                                    library.shape_of_entry(declared_entries[row["entry"]])})

        print(f"    {len(measured_entries)} of {len(run.entry_rows)} row(s) measured a cell; "
              + ("the shapes they measure: "
                 + ", ".join(f"Shape::{shape}" for shape in entry_shapes_book)
                 if entry_shapes_book else "none of them did"))

    if unplaced:
        print(f"  {len(unplaced)} entry(ies) could not be placed: " + ", ".join(sorted(unplaced)))

    # --------------------------------------------------------------------- the gap
    print(f"\nthe gap: every combination the library carries and the recorded run does not measure")

    if not uncovered:
        print("  none")
    else:
        grouped: dict[tuple, list] = {}

        for key in uncovered:
            lane_member, route, scheme, axis, partition, form, exp = key
            lane = next(row for row in library.lanes if row["member"] == lane_member)
            credited = lane_credited[lane["member"]]
            where = []

            if exp not in credited:
                where.append(f"exp={library.names.get(exp, exp)}")

            # The figure the uncovered member's combination is judged by, against the figure
            # the member this lane's cells WERE read at answers - which is the comparison the
            # sentence below the group rests on, and the member it compares against is the
            # credited one rather than the one the lane's row publishes its term under.
            at = credited[0] if credited else lane["additiveMember"]
            same = (library.accessor_figure(lane, "kExactDivision", exp) ==
                    library.accessor_figure(lane, "kExactDivision", at))

            grouped.setdefault((lane["name"], tuple(where), same), []).append(key)

        for (lane_name, where, same), keys in sorted(grouped.items(), key=lambda item: str(item[0])):
            kind = ("the same figure as the measured member - this lane's row publishes no term "
                    "beside the base" if same else
                    "a different figure from the measured member - the bound itself is unmeasured "
                    "here")
            print(f"  {lane_name}, {', '.join(where) if where else '(the measured member)'}: "
                  f"{len(keys)} combination(s) - {kind}")

        print(f"  total: {len(uncovered)} of {accessor_total}")

    if args.full:
        print("\nevery class and every combination")
        for entry in classes:
            for key in entry["space"]:
                row = measured.get(key)
                state = ("measured, state: " + row["state"] + ", bound the run recorded: "
                         + row["bound"]) if row is not None else "NOT MEASURED"
                print(f"  {entry['device'][1:]} {entry['lane']['name']} {entry['shape'][1:]:<14} "
                      f"{', '.join(library.names.get(part, part) for part in key[1:4])}, "
                      f"{library.names.get(key[4], key[4])}, {library.names.get(key[5], key[5])}, "
                      f"{library.names.get(key[6], key[6])}  {state}")

    # ------------------------------------------------------- where the granularities differ
    print("\nwhere the gate's granularity does not match the question")
    print(f"  per lane: the accessor BoysAccuracyGuaranteed takes no Shape, so the figure a class is")
    print(f"    answered with is its lane's. The run's rows are keyed by lane, so a reader comparing")
    print(f"    a class against them cannot tell which shape produced the figure.")
    print(f"  per class: the run's rows measure one entry per lane - "
          f"{', '.join(host_entries) if host_entries else 'none'}"
          + (f" beside the device AllOrders family" if device_entries else "")
          + f". Which classes those entries answer for is read per machine, because an entry "
            f"answers for a class of its own machine and of no other: the host entries answer for "
          + (", ".join(f"Shape::{shape}" for shape in sorted(cross_shapes.get("host", ())))
             if cross_shapes.get("host") else "no shape")
          + f" and the device entries for "
          + (", ".join(f"Shape::{shape}" for shape in sorted(cross_shapes.get("device", ())))
             if cross_shapes.get("device") else "no shape")
          + f", so of {len(classes)} class(es) the book covers "
          f"{sum(1 for e in classes if e['shape_measured'])} and the other "
          f"{sum(1 for e in classes if not e['shape_measured'])} are measured by no row of this "
          f"book.")

    unmeasured_shapes = sorted({entry["shape"] for entry in classes if not entry["shape_measured"]})

    if unmeasured_shapes:
        print(f"    the shapes the combination book does not reach: "
              f"{', '.join('Shape::' + shape for shape in unmeasured_shapes)}")
        print(f"    the entry book reaches some of them, and it is the run's one book that does: it is")
        print(f"    keyed by the entry a caller names, and its rows sit at one lane (the double batch")
        print(f"    lane) and one partition (the uniform one). It does not cross the division form,")
        print(f"    which is the axis the accessor's figures move on, so a class the entry book reaches")
        print(f"    is measured at an entry without being measured at a form.")

    measured_member = ', '.join(sorted({row["additiveMember"] for row in library.lanes}))

    print(f"  per combination: the run measures the division form inside each row and does not cross")
    print(f"    the region-B exponential at all. On a host lane a row's cells are read at every member")
    print(f"    the run's own sentence covers the lane at. On a device lane they are read at the member")
    print(f"    the entries the block arms that lane with run - the table above - because the member is")
    print(f"    the entry's there and no policy argument moves it. The member a lane's row publishes a")
    print(f"    term beside the base under is {measured_member} on every lane, which is where that")
    print(f"    term belongs and not where an entry is.")

    if distinguishing:
        print(f"    the two members answer different figures at one form on: " + "; ".join(distinguishing))
        print(f"    there the unmeasured member is a bound and not a spelling: two figures are two")
        print(f"    claims, and the run carries one of them.")
    else:
        print(f"    on this revision's rows the two members answer one figure on every lane, so the")
        print(f"    bound leg is the same number at both; the arithmetic the other member selects is not.")

    # ---------------------------------------------------------------- compile control
    print("\nthe one-order exclusion: an axis the one-order shapes do not carry")

    one_order_host = [shape for shape in library.shape_order
                      if shape in library.one_order and shape in ("kSingle", "kFixedN")]

    if not args.compile:
        print(f"  asserted from the library's own documentation of "
              f"{', '.join('Shape::' + shape for shape in sorted(library.one_order)) or 'no shape'}, "
              f"and NOT proved by compiling: the control costs minutes of one core per "
              f"instantiation and is off by default")
        print(f"  --compile proves it: one instantiation at PackAxis::{ORDERS_MEMBER} must be "
              f"refused and one at another member accepted, for each one-order shape")
    else:
        entries_by_shape = {}

        for shape in one_order_host:
            name = next((entry for entry, triple in library.entry_class.items()
                         if triple[1] == shape and triple[0] == "kFp64"), None)

            if name is not None:
                entries_by_shape[shape] = name

        if not entries_by_shape:
            findings.append("no double-precision entry of a one-order shape could be read from the "
                            "library, so the exclusion of the orders packing axis rests on the "
                            "enumeration's documentation alone")
            print("  no entry found: the exclusion rests on the documentation alone")
        else:
            with tempfile.TemporaryDirectory() as scratch:
                directory = pathlib.Path(scratch)
                include = materialize(revision, directory)

                if include is None:
                    print(f"  the include tree of {revision} could not be written to a scratch "
                          f"directory, so the exclusion was not put to the compiler: the working "
                          f"tree is what the other lanes hold and is not what this check reads")
                    findings.append(
                        f"the include tree of {revision} could not be materialised, so the "
                        f"one-order exclusion was neither proved nor refuted at this revision")
                else:
                    print(f"  compiled against the include tree of {revision}, written to a "
                          f"scratch directory; the working tree is not read")

                    for shape, entry in sorted(entries_by_shape.items()):
                        at_orders = compiles(unit_for("kFp64", shape, ORDERS_MEMBER,
                                                      entries_by_shape), directory, include)
                        without = compiles(unit_for("kFp64", shape, "kArguments",
                                                    entries_by_shape), directory, include)

                        if at_orders or not without:
                            print(f"  [{shape}] at the orders axis compiles = {at_orders}, without "
                                  f"it compiles = {without} - the exclusion this check rests on does "
                                  f"not hold, so its counts are not admissible")
                            findings.append(
                                f"the one-order exclusion does not hold for Shape::{shape}: an "
                                f"instantiation at PackAxis::{ORDERS_MEMBER} compiled = "
                                f"{at_orders} and one without it compiled = {without}")
                        else:
                            print(f"  [{shape}] the orders axis is refused by the compiler and the "
                                  f"arguments axis is accepted (entry {entry})")

    # ------------------------------------------------------------------------- verdict
    print("\nverdict")

    if findings:
        for finding in findings:
            print(f"  FINDING: {finding}")

    if uncovered:
        print(f"  {len(uncovered)} of {accessor_total} combination(s) the library carries are "
              f"measured by no row of the recorded run.")

    if findings or uncovered:
        if run.device_measured is not None and run.device_measured < run.device_carried:
            print(f"\nFAIL: this record was taken from a build that measured {run.device_measured} "
                  f"of the {run.device_carried} device lane(s) it carries, so it covers the host "
                  f"half and not the whole space. The uncovered combinations above are on the lanes "
                  f"it did not reach, and what is short is the build the record came from and not "
                  f"the library: re-make the run from a build with BUILD_CUDA=ON on a host with a "
                  f"usable card, and commit the new record")
        else:
            print(f"\nFAIL: the recorded run does not cover the combinations this revision carries")
        return 1

    print("  PASS: every combination the library carries is measured by the recorded run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
