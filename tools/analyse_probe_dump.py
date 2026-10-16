#!/usr/bin/env python3
r"""Redoes the option probe's ranking offline, from its raw dump, so the whole ranking analysis can
be re-made without new machine time.

WHY THIS EXISTS. The option probe measures every option of a class in a round of timed visits and
ranks the class. Its report is printed once, by the run, on one machine, and every later question
about that ranking - was the leader really faster, how wide is the band the run could not resolve,
what would a different tie-break have picked - needed the machine again, because the report carries
the answers rather than the measurements they were made from. The raw dump carries the measurements:
one line per timed visit, with the pass, the round, the within-round visit position and the
per-argument cost of the whole timed unit. This file is the analysis layer over that dump, and it is
the whole ranking analysis: the paired log-time fit, the simultaneous intervals, the run's own noise
floor, its minimum detectable effect, the tie set and the pick.

WHAT IT DOES NOT DO.
  * It does not measure anything, and it cannot. Every figure it prints is a function of the dump
    and of the protocol line's own keys. A dump is a measurement of one machine at one moment, and
    a ranking recomputed from it is still that run's ranking, quoted at that run's date.
  * It does not judge the run. Whether the run's passes ran on a steady machine is the probe's own
    canary question, answered by the run's report and not here.
  * It does not compare classes. A class is one precision and one question shape; two classes'
    figures are not the same quantity, and no ordering is formed across them.
  * It does not decide anything by a vote. The class default is named by the declared tie-break
    below and by nothing else.
  * It uses the standard library and nothing else. The design is balanced, so the fit needs a
    de-meaning and one scalar normal equation rather than a matrix package.

THE FIT. `log t ~ option + round + position` is fitted by the within transformation: every log time
is de-meaned by its round, which removes the round effects and is the fixed-effects estimator. For a
balanced panel the within transformation gives the same option estimates as full dummy OLS, because
the two are the same projection written two ways. On the de-meaned data the position covariate is
removed by a second de-meaning, by option, and the slope is then one scalar normal equation through
the origin (the exact Frisch-Waugh step); the option levels are the exact group means of what is
left. One column is solved, by hand, and no matrix is inverted. A visit whose name carries the
`(aa)` prefix is a second measurement of an option already measured in that round and is left out of
the fit, so each option contributes exactly one visit per round. Every interval, every sigma and
every tie is computed from that panel. The residual sigma is `sqrt(SSR / dof)` with
`dof = n - rounds - options`, which is what an option-plus-round-plus-slope model leaves.

An option's fitted level is `mean over its rounds of (round-de-meaned log t) - slope * mean over its
rounds of (round-de-meaned position)`. The intervals are built from the exact variances of those
levels: an option's level variance is `sigma^2 / n_o + pbar_o^2 * sigma^2 / s`, with `pbar_o` its
mean de-meaned position and `s` the panel's spread of de-meaned positions, and the covariance of two
levels is `pbar_i * pbar_j * sigma^2 / s`. The cross terms with the slope vanish exactly rather than
by assumption, because the slope enters each level through a deviation from that option's own mean
position.

THE CLASS RULE, declared here and used as declared: a class is one precision and one question shape,
both encoded in the option's name. The precision is the name's suffix - `-fp64`, `-fp32`, `-fp16` or
`-bf16` - and the shape is the name's leading token set: `single` for a name starting `single-`, else
`all-n-at-orders` for `all-n-at-orders-`, else `all-n` for `all-n-`, else `fixed-n` for `fixed-n-`,
else `all-orders`. The `(aa)` prefix is stripped before this test. A name this rule does not place,
and which carries no class column either, goes into a class named `unclassified`, and a dump with one
exits nonzero: an option nothing can be placed beside is an option no ranking here covers. A class
column (`class=<precision>/<shape>` on a `c` or a `t` line) is preferred to the name rule where a
future dump carries one.

THE INTERVAL RULE, printed beside every interval it produces: Bonferroni-corrected pairwise t
intervals, `alpha = 0.05` over the class's `m * (m - 1) / 2` pairs, so a half width of
`t(1 - alpha / (2 * k), dof) * sigma * se`. The studentized-range quantile that Tukey HSD would need
is not implemented here, and a reader is never left to guess which correction produced a number:
every interval line names this rule and the count it was corrected over.

THE TIE-BREAK, declared here and used as declared, in order: (1) the smaller total table bytes, read
from the option's own byte column where the dump carries one (`bytes=<n>`, or a bare integer after
the repeat count on a `c` line); (2) the lexicographically first name among the tied. Today's dumps
carry no byte column, so clause (1) is unavailable, the report says so, and clause (2) decides. The
pick is a tie-break and not a measurement: where it differs from the fit's fastest option, the report
prints both and the width of the band that could not be resolved.

THE NOISE FLOOR AND THE MDE. The A/A self-pair is the round's own control: the `(aa)` visit is a
second measurement of one option in the same round, so `t(aa) / t(option)` inside that round is the
run's noise floor in the very quantity the ordering is made in, and it is printed before any ordering
claim. The minimum detectable effect at 80% power is
`(1.959964 + 0.841621) * sigma / sqrt(N_eff)`, printed as `(exp(MDE) - 1) * 100`, with sigma the
robust sigma `1.4826 * MAD`, median-centred, of the within-round log-ratios of the class (each option
against the class's own anchor in the same round), N the count of those log-ratios, and
`N_eff = rounds / (1 + (m - 1) * rho)` where rho is the one-way random-effects ICC with the round as
the grouping, `(MSB - MSW) / (MSB + (m - 1) * MSW)`, clamped into `[0, 1)`. That sigma is the spread
of the log-ratios themselves, so a class whose options sit far apart carries its own option-to-option
spacing inside it as well as the noise: it is the spread of the very quantity the effect is quoted
for, which is what an MDE is a statement about, and the A/A band printed above it is the noise of one
visit against another. Both are printed and the report says which is which, so neither is read as the
other.

THE SYNTHETIC DUMP. `--emit-synthetic` writes a small dump of the frozen format whose truth is known
with no machine run, so the analysis can be verified without one. It is one class, `fp64 single`, of
six options at 8.00, 8.15, 8.15, 10.00, 12.00 and 16.00 nanoseconds per argument:
`single-batch-plain-reciprocal-fp64` is the fastest in truth; `single-batch-chebyshev-fp64` and
`single-batch-rational-minimax-fp64` are exactly tied with each other and only 1.9% behind it, which
is inside this run's own resolution; and `single-batch-chebyshev-exact-division-fp64`,
`single-batch-chebyshev-split-clenshaw-fp64` and `single-batch-rational-minimax-fast-fp64` sit 25%,
50% and 100% behind the fastest, which is outside it. Three passes of three rounds carry a per-round
clock effect, a warm-up trend on the visit position and 2% log-normal per-visit noise, all drawn from
`random.Random(47)`, and the A/A column re-measures the anchor once per round. So the reading to
check the analysis against: the fit separates the three slow options; the tie set holds the three
fast ones and no more; the band prints as unresolved, a few percent wide; and the pick is clause
(2)'s - `single-batch-chebyshev-fp64`, which is NOT the option the fit put first, because that is
what a declared tie-break does and what a vote would not. It knows none of this: the truth is a
comment in the file and a line in the report, and the ranking is computed from the visits alone. A
synthetic file is read by the same reader as a real dump, and every section of this report is printed
for it.

Usage:
    python -I tools/analyse_probe_dump.py --dump <dump>
    python -I tools/analyse_probe_dump.py --emit-synthetic <dump>
    python -I tools/analyse_probe_dump.py --dump <dump> --min-class-options 4
    python -I tools/analyse_probe_dump.py --dump <dump> --check

Exit status is 0 when every class of the dump was analysed; 1 when the dump cannot be read, when a
line of it cannot be read, when the format version is not one this knows, when an option had to be
placed in `unclassified`, or when a class was left unanalysed for want of rounds or options. A class
skipped by `--min-class-options` is a class left unanalysed like any other and exits 1 as well: the
exit status is this report's claim about the completeness of the ranking, and a reader who asked for
the skip still sees the reason printed beside the class. A tie set wider than one option is NOT a
failure: it is printed as `unresolved at +/- <width>` beside the pick, which is the honest non-answer
the run's own resolution implies.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import random
import re
import sys
from dataclasses import dataclass, field
from typing import Sequence

# --- the frozen format, version 1 --------------------------------------------

# The format versions this reader knows. A dump that declares another one is refused rather than read
# optimistically: a field added or re-ordered by a later version is a figure misread here, and a
# misread figure printed as a ranking is worse than no ranking at all.
KNOWN_FORMAT_VERSIONS = (1,)

HEADER = re.compile(r"^#\s*boys-option-probe raw dump v(\d+)\s*$")
PROTOCOL_KEY = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
END_LINE = re.compile(r"^#\s*end\s+lines=(\d+)\s*$")

# The prefix of a second measurement of one option in the same round. The A/A visit is a visit like
# any other; it is the name that says it is a re-measurement, and it is kept out of the ranking.
AA = "(aa)"

# The class rule, in the order it is applied. The order matters: `all-n-at-orders-` starts with
# `all-n-`, so the longer token is tested first and no name is classed by the shorter one.
PRECISIONS = (("-fp64", "fp64"), ("-fp32", "fp32"), ("-fp16", "fp16"), ("-bf16", "bf16"))
SHAPES = (("single", "single-"), ("all-n-at-orders", "all-n-at-orders-"), ("all-n", "all-n-"),
          ("fixed-n", "fixed-n-"), ("all-orders", "all-orders"))

# The class a name nothing places goes into, and the words the report prints for it.
UNCLASSIFIED_NAME = "unclassified"

# --- the analysis ------------------------------------------------------------

# The simultaneous level every interval in this report is made at, and the two quantiles the minimum
# detectable effect at 80% power is built from at that level.
ALPHA = 0.05
Z_975 = 1.959964
Z_80 = 0.841621

# The constant that turns a median absolute deviation into a sigma for a normal quantity. Printed
# beside every figure it produced: a robust sigma and a plain standard deviation of the same numbers
# differ, and a reader is owed the one in force.
MAD_CONSTANT = 1.4826

# rho is clamped into [0, 1). The ceiling is just below one so that N_eff, which divides by
# (1 + (m - 1) * rho), stays finite even when every log-ratio of a round moved together.
RHO_CEILING = 1.0 - 1e-6

# The protocol keys this report looks for, so that an absent one is named rather than defaulted in
# silence. None of them is required: a dump with none of them still analyses, and says what it lacked.
EXPECTED_KEYS = ("count", "unit_target_ms", "passes", "rounds", "seed", "reference", "version")


class Unreadable(Exception):
    """A dump this reader cannot read at all: no file, no header, or no version it knows."""


# --- the dump ----------------------------------------------------------------


@dataclass(frozen=True)
class Visit:
    """One timed visit: a whole timed unit's per-argument cost, at one position of one round."""

    pass_no: int
    round_no: int
    position: int
    ns_per_argument: float
    repeats: int
    name: str
    declared: str = ""

    @property
    def round_key(self) -> tuple[int, int]:
        """The round this visit belongs to: one pass and one round number together."""
        return (self.pass_no, self.round_no)

    @property
    def is_aa(self) -> bool:
        return self.name.startswith(AA)

    @property
    def base_name(self) -> str:
        """The name without the A/A prefix, which is the option the visit measures."""
        return self.name[len(AA):] if self.is_aa else self.name


@dataclass(frozen=True)
class Calibration:
    """One `c` line: the probe's own calibration of an option, and the repeats it chose."""

    name: str
    ns_per_argument: float
    repeats: int
    table_bytes: int | None = None
    declared: str = ""


@dataclass
class Dump:
    """A parsed raw dump: what it says about itself, and every line this reader placed."""

    path: pathlib.Path
    header: str
    format_version: int | None
    protocol: dict[str, str]
    protocol_line: str
    notes: list[str]
    calibrations: dict[str, Calibration]
    visits: list[Visit]
    stated_lines: int | None
    malformed: list[tuple[int, str]]
    counted_lines: int
    blank: bool


def parse_dump(path: pathlib.Path) -> Dump:
    """Read a raw dump. Raises `Unreadable` for a file that is not one.

    The header is the first non-empty line and carries the format version; the protocol line is the
    first comment line carrying `key=value` pairs. Any key may be absent, and an absent one is
    reported rather than defaulted silently. A line the reader cannot place is collected and printed
    by the caller, which exits nonzero for it: a dump read in part is a ranking over part of a run,
    and the part that was dropped is exactly what a reader cannot see.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as bad:
        raise Unreadable(f"{path}: {bad.strerror or bad}") from bad

    header = ""
    format_version: int | None = None
    protocol: dict[str, str] = {}
    protocol_line = ""
    notes: list[str] = []
    calibrations: dict[str, Calibration] = {}
    visits: list[Visit] = []
    malformed: list[tuple[int, str]] = []
    stated_lines: int | None = None
    counted = 0
    seen_header = False

    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()

        if not line:
            continue

        if line.startswith("#"):
            found = HEADER.match(line)

            if found is not None and not seen_header:
                seen_header = True
                header = line
                format_version = int(found.group(1))
                continue

            ended = END_LINE.match(line)

            if ended is not None:
                stated_lines = int(ended.group(1))
                continue

            pairs = PROTOCOL_KEY.findall(line)

            if not protocol_line and pairs and ("protocol" in line or len(pairs) > 1):
                protocol_line = line
                protocol = dict(pairs)
                continue

            notes.append(line)
            continue

        counted += 1
        fields = line.split()
        kind = fields[0]

        if kind == "c":
            placed = read_calibration(fields)

            if placed is None:
                malformed.append((number, raw))
            else:
                calibrations[placed.name] = placed
        elif kind == "t":
            visit = read_visit(fields)

            if visit is None:
                malformed.append((number, raw))
            else:
                visits.append(visit)
        else:
            malformed.append((number, raw))

    if not seen_header:
        first = next((line.strip() for line in text.splitlines() if line.strip()), "")
        raise Unreadable(
            f"{path}: the first non-empty line is not the frozen format's header - it reads "
            f"{first[:80]!r}, and a dump is placed by that line rather than guessed at")

    return Dump(path=path, header=header, format_version=format_version, protocol=protocol,
                protocol_line=protocol_line, notes=notes, calibrations=calibrations, visits=visits,
                stated_lines=stated_lines, malformed=malformed, counted_lines=counted,
                blank=not text.strip())


def key_value_extras(fields: Sequence[str]) -> tuple[int | None, str]:
    """The optional trailing fields of a `c` or a `t` line: table bytes, and a class column.

    Today's dump carries neither, and both are read so that a later dump which does carry them is
    placed by this file rather than refused: a bare integer after the repeat count is the option's
    table bytes, `bytes=<n>` is the same figure spelled out, and `class=` is the option's declared
    class. A field of another shape is ignored, and every figure the report prints still comes from
    the frozen columns alone.
    """
    table_bytes: int | None = None
    declared = ""

    for extra in fields:
        if extra.startswith("class="):
            declared = extra[len("class="):]
        elif extra.startswith("bytes="):
            try:
                table_bytes = int(extra[len("bytes="):])
            except ValueError:
                table_bytes = None
        elif extra.isdigit() and table_bytes is None:
            table_bytes = int(extra)

    return table_bytes, declared


def read_calibration(fields: Sequence[str]) -> Calibration | None:
    """One `c` line: `c <name> <one_call_ns_per_arg> <repeats>` and any optional extras."""
    if len(fields) < 4:
        return None

    try:
        ns_per_argument = float(fields[2])
        repeats = int(fields[3])
    except ValueError:
        return None

    table_bytes, declared = key_value_extras(fields[4:])

    return Calibration(name=fields[1], ns_per_argument=ns_per_argument, repeats=repeats,
                       table_bytes=table_bytes, declared=declared)


def read_visit(fields: Sequence[str]) -> Visit | None:
    """One `t` line: `t <pass> <round> <position> <ns> <repeats> <name>` and optional extras."""
    if len(fields) < 7:
        return None

    try:
        pass_no = int(fields[1])
        round_no = int(fields[2])
        position = int(fields[3])
        ns_per_argument = float(fields[4])
        repeats = int(fields[5])
    except ValueError:
        return None

    _table_bytes, declared = key_value_extras(fields[7:])

    return Visit(pass_no=pass_no, round_no=round_no, position=position,
                 ns_per_argument=ns_per_argument, repeats=repeats, name=fields[6], declared=declared)


def normalise(token: str) -> str:
    return " ".join(token.strip().lower().replace("_", "-").split())


def precision_of(name: str) -> str:
    """The precision a name's suffix declares, or an empty string where it declares none."""
    base = name[len(AA):] if name.startswith(AA) else name

    for suffix, precision in PRECISIONS:
        if base.endswith(suffix):
            return precision

    return ""


def shape_of(name: str) -> str:
    """The question shape a name's leading token set declares, or an empty string."""
    base = name[len(AA):] if name.startswith(AA) else name

    for shape, prefix in SHAPES:
        if base.startswith(prefix):
            return shape

    return ""


def parse_declared_class(raw: str, name: str) -> tuple[str, str] | None:
    """A class column's value, as a (precision, shape) pair, or None where it cannot be placed.

    The column is written `precision/shape` or `precision:shape`; a bare shape takes its precision
    from the name's suffix, so a dump that carries only the shape still places its options.
    """
    for separator in ("/", ":"):
        if separator in raw:
            precision, _, shape = raw.partition(separator)

            return (normalise(precision), normalise(shape)) if precision and shape else None

    return (precision_of(name), normalise(raw)) if precision_of(name) else None


def classify(name: str, declared: dict[str, tuple[str, str]]) -> tuple[str, str] | None:
    """The class of an option: the declared column where the dump carries one, else the name rule.

    None is the answer for a name the rule does not place. It is not a class, and the option is
    reported unclassified rather than filed under a class it resembles.
    """
    base = name[len(AA):] if name.startswith(AA) else name

    if base in declared:
        return declared[base]

    precision, shape = precision_of(name), shape_of(name)

    return (precision, shape) if precision and shape else None


def declared_classes(dump: Dump) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """Every class column the dump carries, as option -> class, and the conflicts it carries.

    A class column wins over the name rule where both place an option, and the two disagreeing is
    printed: a name and a column that say different things about one option mean one of them is
    wrong, and this reader will not pick silently between them.
    """
    declared: dict[str, tuple[str, str]] = {}
    conflicts: list[str] = []

    for name, raw in [(option, entry.declared) for option, entry in dump.calibrations.items()] \
            + [(visit.base_name, visit.declared) for visit in dump.visits if visit.declared]:
        if not raw:
            continue

        placed = parse_declared_class(raw, name)

        if placed is None:
            conflicts.append(f"{name}: the class column reads {raw!r}, which this reader cannot "
                             f"place, so the name rule was used instead")
            continue

        previous = declared.get(name)

        if previous is not None and previous != placed:
            conflicts.append(f"{name}: the dump declares two classes, {display_class(previous)} "
                             f"and {display_class(placed)}; the first was used")
            continue

        declared[name] = placed

    return declared, conflicts


def display_class(key: tuple[str, str]) -> str:
    """A class as the report prints it: the precision and the shape."""
    precision, shape = key

    return f"{precision} {shape}" if shape else (precision or UNCLASSIFIED_NAME)


# --- the statistics ----------------------------------------------------------


def log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def beta_continued_fraction(a: float, b: float, x: float) -> float:
    """The continued fraction of the incomplete beta, by the modified Lentz method."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap

    if abs(d) < tiny:
        d = tiny

    d = 1.0 / d
    h = d

    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d

        if abs(d) < tiny:
            d = tiny

        c = 1.0 + aa / c

        if abs(c) < tiny:
            c = tiny

        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d

        if abs(d) < tiny:
            d = tiny

        c = 1.0 + aa / c

        if abs(c) < tiny:
            c = tiny

        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < 3e-16:
            break

    return h


def incomplete_beta(a: float, b: float, x: float) -> float:
    """The regularized incomplete beta `I_x(a, b)`, the function the t distribution is written in."""
    if x <= 0.0:
        return 0.0

    if x >= 1.0:
        return 1.0

    front = math.exp(a * math.log(x) + b * math.log1p(-x) - log_beta(a, b))

    if x < (a + 1.0) / (a + b + 2.0):
        return front * beta_continued_fraction(a, b, x) / a

    return 1.0 - math.exp(b * math.log1p(-x) + a * math.log(x) - log_beta(a, b)) \
        * beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_cdf(t: float, dof: float) -> float:
    """P(T <= t) for a Student t with `dof` degrees of freedom."""
    if dof <= 0.0:
        return float("nan")

    tail = 0.5 * incomplete_beta(0.5 * dof, 0.5, dof / (dof + t * t))

    return 1.0 - tail if t >= 0.0 else tail


def student_t_quantile(p: float, dof: float) -> float:
    """The t with P(T <= t) = p, by bisection on the distribution function.

    The studentized-range quantile Tukey HSD needs is not implemented here; this is the t the
    Bonferroni correction uses, and every interval line of the report names that rule.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"a quantile needs a probability inside (0, 1), not {p}")

    if p == 0.5:
        return 0.0

    if p < 0.5:
        return -student_t_quantile(1.0 - p, dof)

    low, high = 0.0, 1.0

    while student_t_cdf(high, dof) < p and high < 1e8:
        high *= 2.0

    for _ in range(200):
        middle = 0.5 * (low + high)

        if student_t_cdf(middle, dof) < p:
            low = middle
        else:
            high = middle

        if high - low <= 1e-12 * max(1.0, high):
            break

    return 0.5 * (low + high)


def median(values: Sequence[float]) -> float:
    """The median of a sequence, or NaN where it holds nothing."""
    ordered = sorted(values)
    count = len(ordered)

    if count == 0:
        return float("nan")

    middle = count // 2

    return ordered[middle] if count % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])


def robust_sigma(values: Sequence[float]) -> float:
    """`1.4826 * MAD`, median-centred: a robust sigma for a normal quantity.

    A median-centred MAD leaves the median of the numbers out of the figure, which is what makes it
    usable on the within-round log-ratios here: those are all measured from one anchor option, so
    their median is the anchor's own distance from the class and not a noise at all.
    """
    if not values:
        return float("nan")

    centre = median(values)

    return MAD_CONSTANT * median([abs(value - centre) for value in values])


def one_way_icc(groups: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    """The one-way random-effects ICC of grouped values, with the mean squares it was made from.

    Returns (rho, MSB, MSW, m0). `m0` is the group size the unbalanced form of the estimator uses,
    and it is the common group size when every group is the same size. The groups here are the
    rounds, and the values are that class's log-ratios, so rho is the share of a log-ratio's spread
    that a round's own clock moved rather than the two visits differing.
    """
    k = len(groups)
    sizes = [len(group) for group in groups]
    total = sum(sizes)

    if k < 2 or total - k <= 0:
        return (float("nan"), 0.0, 0.0, float(sizes[0]) if sizes else 0.0)

    overall = sum(sum(group) for group in groups) / total
    ssb = sum(size * (sum(group) / size - overall) ** 2 for size, group in zip(sizes, groups))
    ssw = sum(sum((value - sum(group) / size) ** 2 for value in group)
              for size, group in zip(sizes, groups))
    msb = ssb / (k - 1)
    msw = ssw / (total - k)
    m0 = (total - sum(size * size for size in sizes) / total) / (k - 1)
    denominator = msb + (m0 - 1.0) * msw

    if denominator <= 0.0 or math.isnan(denominator):
        return (0.0, msb, msw, m0)

    return ((msb - msw) / denominator, msb, msw, m0)


# --- the fit -----------------------------------------------------------------


@dataclass
class Fit:
    """One class's paired log-time fit, and everything an interval over it needs."""

    options: list[str]
    levels: dict[str, float]
    counts: dict[str, int]
    pbar: dict[str, float]
    table_bytes: dict[str, int | None]
    slope: float
    sigma: float
    dof: int
    observations: int
    rounds: int
    centre: float
    spread: float
    balance: str

    def variance_factor(self, option: str) -> float:
        """`Var(level) / sigma^2` for one option: its own cell mean, plus the slope's correction."""
        correction = self.pbar[option] ** 2 / self.spread if self.spread > 0.0 else 0.0

        return 1.0 / self.counts[option] + correction

    def difference_factor(self, one: str, other: str) -> float:
        """`Var(level_one - level_other) / sigma^2`, exactly, for the fitted panel.

        The slope is estimated from the same log times, so two levels are not independent; the
        correction is the squared difference of their mean de-meaned positions over the panel's own
        spread of positions. The slope's two covariance terms with a level vanish exactly rather than
        by assumption, because it enters that level through a deviation from the option's own mean
        position and such a deviation sums to zero over each option.
        """
        correction = (self.pbar[one] - self.pbar[other]) ** 2 / self.spread \
            if self.spread > 0.0 else 0.0

        return 1.0 / self.counts[one] + 1.0 / self.counts[other] + correction


def fit_class(cells: dict[tuple[tuple[int, int], str], tuple[float, float]],
              options: Sequence[str], rounds: Sequence[tuple[int, int]],
              table_bytes: dict[str, int | None]) -> Fit:
    """Fit `log t ~ option + round + position` on one class by the within transformation.

    Round means come out of every log time and every position first, which is the fixed-effects
    transformation: the fitted values are then the ones full dummy OLS gives on a balanced panel,
    because the two are one projection written two ways. The position slope is removed next by a
    second de-meaning, by option, which is the exact Frisch-Waugh step, so the slope is one scalar
    normal equation through the origin and each option's level is the group mean of what is left.
    """
    present = [key for key in rounds
               if any((key, option) in cells for option in options)]
    counts: dict[str, int] = {option: 0 for option in options}
    position_sum = {option: 0.0 for option in options}
    level_sum = {option: 0.0 for option in options}
    round_level: dict[tuple[int, int], float] = {}
    round_position: dict[tuple[int, int], float] = {}

    for key in present:
        inside = [cells[(key, option)] for option in options if (key, option) in cells]
        round_level[key] = sum(log_t for log_t, _position in inside) / len(inside)
        round_position[key] = sum(position for _log_t, position in inside) / len(inside)

    residual_y: dict[tuple[tuple[int, int], str], float] = {}
    residual_p: dict[tuple[tuple[int, int], str], float] = {}

    for (key, option), (log_t, position) in cells.items():
        residual_y[(key, option)] = log_t - round_level[key]
        residual_p[(key, option)] = position - round_position[key]
        counts[option] += 1
        position_sum[option] += residual_p[(key, option)]
        level_sum[option] += residual_y[(key, option)]

    pbar = {option: position_sum[option] / counts[option] for option in options}
    ybar = {option: level_sum[option] / counts[option] for option in options}
    numerator = sum((residual_y[cell] - ybar[cell[1]]) * (residual_p[cell] - pbar[cell[1]])
                    for cell in cells)
    spread = sum((residual_p[cell] - pbar[cell[1]]) ** 2 for cell in cells)
    slope = numerator / spread if spread > 0.0 else 0.0
    levels = {option: ybar[option] - slope * pbar[option] for option in options}

    ssr = sum((residual_y[cell] - levels[cell[1]] - slope * residual_p[cell]) ** 2
              for cell in cells)
    observations = sum(counts.values())
    dof = observations - len(present) - len(options)
    sigma = math.sqrt(ssr / dof) if dof > 0 else 0.0
    centre = sum(log_t for log_t, _position in cells.values()) / observations

    return Fit(options=list(options), levels=levels, counts=counts, pbar=pbar,
               table_bytes=table_bytes, slope=slope, sigma=sigma, dof=dof,
               observations=observations, rounds=len(present), centre=centre, spread=spread,
               balance=("balanced: every option is visited once in every round"
                        if all(count == len(present) for count in counts.values())
                        else "unbalanced: at least one option is missing from at least one round"))


# --- log-ratios, the noise floor and the MDE ---------------------------------


@dataclass
class RatioPanel:
    """One class's within-round log-ratios: every option against the class's anchor."""

    anchor: str
    anchor_from: str
    values: list[float]
    groups: list[list[float]]
    rounds: list[tuple[int, int]]

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def per_round(self) -> float:
        return len(self.values) / len(self.groups) if self.groups else 0.0


def ratio_panel(visits: Sequence[Visit], anchor: str, options: Sequence[str]) \
        -> tuple[RatioPanel, int]:
    """A class's within-round log-ratios against `anchor`, and how many rounds it was absent from.

    One log-ratio per (round, option): the option's visit against the anchor's visit in the same
    round, so the round's own clock effect cancels out of every number. The A/A visit is left out
    here for the reason it is left out of the fit: it is a second measurement of an option already in
    the panel, and its own use is the noise floor.
    """
    wanted = set(options)
    by_round: dict[tuple[int, int], dict[str, float]] = {}

    for visit in visits:
        if not visit.is_aa and visit.base_name in wanted:
            by_round.setdefault(visit.round_key, {})[visit.base_name] = \
                math.log(visit.ns_per_argument)

    values: list[float] = []
    groups: list[list[float]] = []
    rounds: list[tuple[int, int]] = []
    absent = 0

    for key in sorted(by_round):
        measured = by_round[key]

        if anchor not in measured:
            absent += 1
            continue

        group = [measured[option] - measured[anchor] for option in options
                 if option != anchor and option in measured]

        if not group:
            absent += 1
            continue

        values.extend(group)
        groups.append(group)
        rounds.append(key)

    return RatioPanel(anchor=anchor, anchor_from="", values=values, groups=groups,
                      rounds=rounds), absent


def anchor_for(visits: Sequence[Visit], options: Sequence[str], reference: str) -> tuple[str, str]:
    """A class's anchor and where it came from: the protocol's reference, else the fastest option.

    The anchor is only the origin the log-ratios are measured from, so it is stated rather than
    assumed: the run's own `reference=` where the class carries that option, and otherwise the
    class's own fastest option by mean log time, with the lexicographically first name breaking a tie
    between two identical means.
    """
    if reference and reference in options:
        return reference, "the protocol line's own reference="

    mean_log: dict[str, list[float]] = {option: [] for option in options}

    for visit in visits:
        if not visit.is_aa and visit.base_name in mean_log:
            mean_log[visit.base_name].append(math.log(visit.ns_per_argument))

    scored = sorted(mean_log.items(),
                    key=lambda item: (sum(item[1]) / len(item[1]), item[0]) if item[1]
                    else (float("inf"), item[0]))

    return (scored[0][0], "the class's own fastest option by mean log time") if scored else \
        ("", "nothing: the class carries no visit")


# --- one class's analysis ----------------------------------------------------


@dataclass
class ClassReport:
    """Everything one class's analysis produced, so the report is printed from one place."""

    key: tuple[str, str]
    options: list[str]
    visits: list[Visit]
    rounds: list[tuple[int, int]]
    skipped: str = ""
    fit: Fit | None = None
    panel: RatioPanel | None = None
    absent_rounds: int = 0
    best: str = ""
    pairs: int = 0
    quantile: float = float("nan")
    tie: list[str] = field(default_factory=list)
    pick: str = ""
    clause: str = ""
    clause_one: str = ""
    band: float = float("nan")
    band_note: str = ""


def analyse_class(key: tuple[str, str], options: Sequence[str], visits: Sequence[Visit],
                  reference: str, minimum: int, calibrations: dict[str, Calibration]) -> \
        ClassReport:
    """Analyse one class: the fit, the interval rule, the tie set and the pick."""
    rounds = sorted({visit.round_key for visit in visits})
    report = ClassReport(key=key, options=sorted(options), visits=list(visits), rounds=rounds)

    anchor, anchor_from = anchor_for(visits, report.options, reference)
    panel, absent = ratio_panel(visits, anchor, report.options) if anchor else (None, 0)

    if panel is not None:
        panel.anchor_from = anchor_from

    report.panel, report.absent_rounds = panel, absent

    if len(options) < 2:
        report.skipped = "1 option, and an ordering needs two"
        return report

    if len(options) < minimum:
        report.skipped = (f"{len(options)} option(s), below the {minimum} this run was asked for "
                          f"by --min-class-options")
        return report

    if len(rounds) < 2:
        report.skipped = f"{len(rounds)} round(s), and the fit needs two"
        return report

    cells: dict[tuple[tuple[int, int], str], tuple[float, float]] = {}
    duplicates = 0

    for visit in visits:
        if visit.is_aa:
            continue

        cell = (visit.round_key, visit.base_name)

        if cell in cells:
            duplicates += 1
            continue

        cells[cell] = (math.log(visit.ns_per_argument), float(visit.position))

    if duplicates:
        report.skipped = (f"the panel carries {duplicates} duplicate (round, option) cell(s), "
                          f"which is a dump this reader will not fit over")
        return report

    table_bytes = {option: (calibrations[option].table_bytes
                            if option in calibrations else None) for option in report.options}
    fit = fit_class(cells, report.options, rounds, table_bytes)
    report.fit = fit

    if fit.dof <= 0:
        report.skipped = (f"{fit.dof} degree(s) of freedom left after the rounds and the options, "
                          f"so no residual sigma exists to build an interval from")
        return report

    report.best = min(fit.levels, key=lambda option: (fit.levels[option], option))
    report.pairs = len(report.options) * (len(report.options) - 1) // 2
    report.quantile = student_t_quantile(1.0 - ALPHA / (2.0 * report.pairs), float(fit.dof))
    report.tie = tied_with_best(fit, report.best, report.quantile)
    report.pick, report.clause, report.clause_one = pick_of(report.tie, fit)
    report.band, report.band_note = unresolved_band(fit, report.tie, report.pick, report.quantile)

    return report


def tied_with_best(fit: Fit, best: str, quantile: float) -> list[str]:
    """Every option whose simultaneous interval overlaps the best option's.

    Each option's interval is the one for its difference from the best, which is the quantity the
    ordering is made in, so the best's own interval is the point zero and the test is whether an
    option's interval carries zero inside it.
    """
    tied = []

    for option in fit.options:
        difference = fit.levels[option] - fit.levels[best]
        half = quantile * fit.sigma * math.sqrt(fit.difference_factor(option, best))

        if difference - half <= 0.0:
            tied.append(option)

    return sorted(tied, key=lambda option: (fit.levels[option], option))


def pick_of(tied: Sequence[str], fit: Fit) -> tuple[str, str, str]:
    """The class default by the declared tie-break, the clause that chose it, and clause (1)'s fate.

    Clause (1) is the smaller total table bytes, where the dump carries a byte column; clause (2) is
    the lexicographically first name among the tied. Nothing else is consulted and no vote is taken:
    a pick formed by a rule a reader can re-run is a pick a reader can check.
    """
    if not tied:
        return "", "nothing: the tie set is empty, which is a class with no option in it", ""

    with_bytes = [option for option in tied if fit.table_bytes.get(option) is not None]

    if not with_bytes:
        clause_one = "unavailable - the dump carries no byte column"
    elif len(with_bytes) != len(tied):
        clause_one = (f"unavailable - {len(tied) - len(with_bytes)} of the {len(tied)} tied "
                      f"option(s) carry no byte figure")
    else:
        smallest = min(fit.table_bytes[option] for option in tied)
        holders = [option for option in tied if fit.table_bytes[option] == smallest]

        if len(holders) == 1:
            return (holders[0], "clause (1): the smaller total table bytes",
                    f"available and decisive: {holders[0]} holds {smallest} byte(s)")

        clause_one = f"available but not decisive: {len(holders)} tied option(s) hold {smallest}"

    return (sorted(tied)[0], "clause (2): the lexicographically first name among the tied",
            clause_one)


def unresolved_band(fit: Fit, tied: Sequence[str], pick: str, quantile: float) \
        -> tuple[float, str]:
    """The width the pick could not resolve: the widest tied upper end over the pick's own lower end.

    Both ends are the run's own, in nanoseconds per argument at the declared simultaneous level, so
    the band is the run's honest non-answer: inside it the ordering is not a resolution but noise. A
    band that comes out negative is printed as such with the reason, because that is the one shape
    the reader must not read as a width.
    """
    if len(tied) < 2 or not pick:
        return float("nan"), ""

    others = [option for option in tied if option != pick]

    if not others:
        return float("nan"), ""

    levels = {option: math.exp(fit.centre + fit.levels[option]) for option in fit.options}
    half = {option: quantile * fit.sigma * math.sqrt(fit.variance_factor(option))
            for option in fit.options}
    lower = {option: levels[option] * math.exp(-half[option]) for option in fit.options}
    upper = {option: levels[option] * math.exp(half[option]) for option in fit.options}

    if lower[pick] <= 0.0:
        return float("nan"), ""

    widest = max(upper[option] for option in others)
    band = (widest / lower[pick] - 1.0) * 100.0
    note = "" if band >= 0.0 else (
        "note: negative, the widest tied option's upper end lies below the pick's own lower end, so "
        "the order inside the tie set is not resolved either")

    return band, note


# --- the A/A noise floor -----------------------------------------------------


def a_a_ratios(visits: Sequence[Visit], declared: dict[str, tuple[str, str]]) \
        -> tuple[list[tuple[tuple[int, int], float, str, tuple[str, str]]], list[str]]:
    """Every round's A/A ratio, as (round, ratio, option, its class), and what could not be formed.

    The ratio is taken inside one (pass, round) and nowhere else: it is a second measurement of an
    option against the first, in the same round, so the round's own clock effect cancels and what is
    left is the noise in the very quantity the ordering is made in.
    """
    by_round: dict[tuple[int, int], list[Visit]] = {}

    for visit in visits:
        by_round.setdefault(visit.round_key, []).append(visit)

    ratios: list[tuple[tuple[int, int], float, str, tuple[str, str]]] = []
    notes: list[str] = []

    for key in sorted(by_round):
        inside = by_round[key]
        reported = [visit for visit in inside if visit.is_aa]

        if len(reported) != 1:
            notes.append(f"round {key[0]}/{key[1]} carries {len(reported)} A/A visit(s), and the "
                         f"frozen format carries exactly one")
            continue

        visit = reported[0]
        partner = next((other for other in inside
                        if not other.is_aa and other.base_name == visit.base_name), None)

        if partner is None:
            notes.append(f"round {key[0]}/{key[1]} re-measures {visit.base_name}, which that round "
                         f"does not measure on its own")
            continue

        if partner.ns_per_argument <= 0.0 or visit.ns_per_argument <= 0.0:
            notes.append(f"round {key[0]}/{key[1]} carries a non-positive time")
            continue

        placed = classify(visit.base_name, declared)

        if placed is None:
            notes.append(f"round {key[0]}/{key[1]} re-measures {visit.base_name}, which no class "
                         f"places, so its ratio is reported in the overall band only")
            continue

        ratios.append((key, visit.ns_per_argument / partner.ns_per_argument, visit.base_name,
                       placed))

    return ratios, notes


def band_text(values: Sequence[float]) -> str:
    """A ratio band as the run's own resolution: the ratios, and the same band in percent."""
    if not values:
        return "no pair"

    low, high = min(values), max(values)

    return (f"{low:.4f} .. {high:.4f}, that is {(low - 1.0) * 100.0:+.2f}% .. "
            f"{(high - 1.0) * 100.0:+.2f}%")


# --- printing ----------------------------------------------------------------


def emit_header(dump: Dump, known: bool) -> None:
    """The dump's own statement of itself, before any figure is computed from it."""
    version = "unknown" if dump.format_version is None else str(dump.format_version)

    print("boys option probe raw dump: the ranking analysis, redone offline")
    print(f"  file      {dump.path}")
    print(f"  format    {dump.header if dump.header else 'no header line'}")
    print(f"            version {version} "
          f"({'a version this reads' if known else 'NOT a version this reads'})")
    default_protocol = ("(none: no comment line of this dump carries key=value pairs)")
    print(f"  protocol  {dump.protocol_line if dump.protocol_line else default_protocol}")

    if dump.protocol:
        print("  keys      " + " ".join(f"{key}={value}"
                                        for key, value in sorted(dump.protocol.items())))

    absent = [key for key in EXPECTED_KEYS if key not in dump.protocol]
    print("  absent    " + (", ".join(absent) if absent else
                            "(none of the keys this reader looks for)"))

    stated = dump.stated_lines if dump.stated_lines is not None else "(none)"
    print(f"  counts    the end line states {stated}, the file carries {dump.counted_lines} "
          f"non-comment line(s)")
    print()


def emit_notes(dump: Dump, conflicts: Sequence[str]) -> None:
    """The dump's other comment lines, and the class-column conflicts, verbatim."""
    for note in dump.notes:
        print(f"  {note}")

    if dump.notes:
        print()

    for conflict in conflicts:
        print(f"  class column: {conflict}")

    if conflicts:
        print()


def emit_resolution(ratios: Sequence[tuple[tuple[int, int], float, str, tuple[str, str]]],
                    class_ratios: dict[tuple[str, str], list[float]],
                    panels: dict[tuple[str, str], RatioPanel],
                    skipped: dict[tuple[str, str], str]) -> None:
    """The run's own resolution: the A/A noise floor, then the minimum detectable effect."""
    print("THE RUN'S RESOLUTION - printed before any ordering claim")
    print()
    print("  the A/A self-pair: a second measurement of one option in the same round, so the ratio")
    print("  of the two is that round's noise floor in the quantity the ordering is made in")
    overall = [ratio for _key, ratio, _name, _class in ratios]

    if overall:
        print(f"    overall: {len(overall)} pair(s), median {median(overall):.4f}, band "
              f"{band_text(overall)}")
    else:
        print("    overall: no A/A pair in this dump, so the run states no noise floor of its own")

    print("    per class, the same quantity one class at a time:")
    print(f"      {'class':<24}{'pairs':>7}  {'median':>8}  band")

    for key in sorted(set(class_ratios) | set(skipped)):
        values = class_ratios.get(key, [])

        if values:
            print(f"      {display_class(key):<24}{len(values):>7}  {median(values):>8.4f}  "
                  f"{band_text(values)}")
        else:
            print(f"      {display_class(key):<24}{0:>7}  {'-':>8}  no A/A pair in this class")

    print()
    print("    the minimum detectable effect at 80% power")
    print(f"    sigma = {MAD_CONSTANT} * MAD of the within-round log-ratios, median-centred (a "
          f"robust sigma);")
    print(f"    MDE = ({Z_975} + {Z_80}) * sigma / sqrt(N_eff), reported as (exp(MDE) - 1) * 100.")
    print("    The log-ratios are every option against the class's anchor in the same round; "
          "N_eff =")
    print("    rounds / (1 + (m - 1) * rho), rho the one-way ICC by round, clamped into [0, 1).")
    print("    That sigma is the spread of those log-ratios, so a class whose options sit far apart")
    print("    carries that spacing inside it as well as the noise: the A/A band above is the noise")
    print("    of one visit against another, and both figures are printed, neither read as the "
          "other.")
    print(f"      {'class':<24}{'N':>7}{'m':>8}{'rho':>9}{'N_eff':>9}{'sigma':>9}{'MDE':>10}")

    for key in sorted(panels):
        line = mde_columns(panels[key])

        if key in skipped:
            line += f"   (skipped: {skipped[key]})"

        print(f"      {display_class(key):<24}{line}")

    print()


def mde_columns(panel: RatioPanel) -> str:
    """The N, m, rho, N_eff, sigma and MDE of one class's log-ratio panel, as one row."""
    count = panel.count

    if not panel.groups:
        return (f"{count:>7}{'-':>8}{'-':>9}{'-':>9}{'-':>9}{'n/a':>10}   "
                f"(no within-round log-ratio in this class, so there is no spread to take)")

    sigma = robust_sigma(panel.values)
    rho, _msb, _msw, m0 = one_way_icc(panel.groups)
    rounds = len(panel.groups)
    per_round = m0 if not math.isnan(m0) else panel.per_round

    if math.isnan(rho) or per_round <= 0.0:
        return (f"{count:>7}{per_round:>8.0f}{'-':>9}{float(count):>9.2f}{sigma:>9.4f}"
                f"{mde_percent(sigma, float(count)):>10}   (too few rounds to estimate rho, so "
                f"N_eff = N)")

    clamped = min(max(rho, 0.0), RHO_CEILING)
    effective = rounds / (1.0 + (per_round - 1.0) * clamped)

    if clamped == rho:
        mark = ""
    elif rho < 0.0:
        mark = f"   (rho = {rho:+.4f} is below 0 and is clamped to 0)"
    else:
        mark = f"   (rho = {rho:.4f} reaches the ceiling and is clamped just below 1)"

    thin = "" if count >= 10 else "   (thin: fewer than 10 log-ratios)"

    return (f"{count:>7}{per_round:>8.2f}{clamped:>9.4f}{effective:>9.2f}{sigma:>9.4f}"
            f"{mde_percent(sigma, effective):>10}{mark}{thin}")


def mde_percent(sigma: float, effective: float) -> str:
    """The minimum detectable effect at 80% power, as the relative percentage."""
    if not effective > 0.0 or math.isnan(sigma):
        return "n/a"

    mde = (Z_975 + Z_80) * sigma / math.sqrt(effective)

    return f"{(math.exp(mde) - 1.0) * 100.0:.2f}%"


def report_class(report: ClassReport, calibrations: dict[str, Calibration]) -> None:
    """One class's whole analysis: the fit, the interval rule, the tie set and the pick."""
    print(f"  {display_class(report.key)}: {len(report.options)} option(s), {len(report.rounds)} "
          f"round(s)")

    if report.skipped:
        print(f"    SKIPPED: {report.skipped}")
        print()
        return

    fit = report.fit

    if fit is None:
        print("    SKIPPED: the fit produced nothing, which is not a state this analysis has")
        print()
        return

    anchor = report.panel.anchor if report.panel is not None else ""

    if report.panel is not None:
        print(f"    the anchor of this class's log-ratios: {anchor} "
              f"({report.panel.anchor_from})")

    print("    the paired log-time fit: log t ~ option + round + position")
    print("      the within transformation de-means every log time by its round, which is the")
    print("      fixed-effects estimator; on a balanced panel it gives the same option estimates")
    print(f"      as full dummy OLS. {fit.balance}.")
    print("      the position slope is then solved as ONE scalar normal equation through the")
    print("      origin on the doubly de-meaned data (the exact Frisch-Waugh step), and the option")
    print("      levels are the group means of what is left: 1 column solved, no matrix inverted.")
    print(f"    residual sigma {fit.sigma:.4f} (log units) over dof {fit.dof}, from "
          f"{fit.observations} observation(s) in {fit.rounds} round(s)")
    print(f"    position slope {fit.slope:+.6f} per within-round visit position")
    print()
    print(f"    the fitted effects, relative to the class's best by the fit:")
    width = max([len(option) for option in fit.options] + [len("option")])
    print(f"      {'option':<{width}}  {'fit ns/arg':>10}  {'effect':>9}  {'lo%':>9}  {'hi%':>9}  "
          f"{'cal ns/arg':>10}  {'repeats':>7}")

    for option in sorted(fit.options, key=lambda name: (fit.levels[name], name)):
        difference = fit.levels[option] - fit.levels[report.best]
        half = report.quantile * fit.sigma * math.sqrt(
            fit.difference_factor(option, report.best))
        fitted = math.exp(fit.centre + fit.levels[option])
        calibration = calibrations.get(option)
        calibrated = f"{calibration.ns_per_argument:>10.3f}" if calibration else f"{'-':>10}"
        repeats = f"{calibration.repeats:>7}" if calibration else f"{'-':>7}"

        print(f"      {option:<{width}}  {fitted:>10.3f}  "
              f"{(math.exp(difference) - 1.0) * 100.0:>8.2f}%  "
              f"{(math.exp(difference - half) - 1.0) * 100.0:>8.2f}%  "
              f"{(math.exp(difference + half) - 1.0) * 100.0:>8.2f}%  {calibrated}  {repeats}")

    print()
    print(f"    the {100 * (1 - ALPHA):.0f}% simultaneous intervals")
    print(f"      rule: Bonferroni-corrected pairwise t intervals, alpha = {ALPHA} over "
          f"{report.pairs} pair(s),")
    print(f"      so t(1 - {ALPHA} / (2 * {report.pairs}), {fit.dof}) = {report.quantile:.4f} on "
          f"the residual sigma above.")
    print(f"      Each row's interval is on its difference from {report.best}, which is the")
    print("      quantity the ordering is made in; the studentized-range quantile Tukey HSD needs")
    print("      is not implemented here, so this rule and not that one made every interval above.")
    print()
    print(f"    the tie set: {len(report.tie)} of {len(fit.options)} option(s) whose interval "
          f"overlaps {report.best}'s")

    for option in report.tie:
        print(f"      {option}")

    print()
    print("    the pick")
    print(f"      the fastest by the fit: {report.best}")
    print(f"      clause (1) the smaller total table bytes: {report.clause_one}")
    print(f"      clause (2) the lexicographically first name among the tied")
    print(f"      pick: {report.pick}")
    print(f"      chosen by {report.clause}")

    if len(report.tie) > 1:
        print(f"      unresolved at +/- {report.band:.2f}%: the widest tied option's own level")
        print(f"      interval upper end over the pick's own level interval lower end")

        if report.band_note:
            print(f"      {report.band_note}")

        if report.best != report.pick:
            print("      the pick is not the fit's fastest option: the two are inside one tie set,")
            print("      and the tie-break chose between them by its declared clause and not by the")
            print("      fit")
    else:
        print(f"      resolved: the tie set holds the pick alone, at "
              f"{math.exp(fit.centre + fit.levels[report.pick]):.3f} ns/argument")

    print()


# --- the synthetic dump ------------------------------------------------------


# The synthetic run's own truth. The generator writes it into the file as comments and into the
# report as lines, and the analysis above is never told any of it: the ranking comes from the visits.
# The six factors are the true nanosecond cost of one argument divided by SYNTHETIC_BASE_NS: the first
# option is fastest, the second and third are exactly tied with each other and just inside the run's
# resolution of the first, and the last three are far enough behind to be separated by it.
SYNTHETIC_SEED = 47
SYNTHETIC_PASSES = 3
SYNTHETIC_ROUNDS = 3
SYNTHETIC_BASE_NS = 10.0
SYNTHETIC_NOISE = 0.02
SYNTHETIC_ROUND_NOISE = 0.010
SYNTHETIC_POSITION_SLOPE = 0.002
SYNTHETIC_COUNT = 16384
SYNTHETIC_UNIT_TARGET_MS = 6.0
SYNTHETIC_ANCHOR = "single-batch-plain-reciprocal-fp64"
SYNTHETIC_OPTIONS = (("single-batch-plain-reciprocal-fp64", 0.800),
                     ("single-batch-chebyshev-fp64", 0.815),
                     ("single-batch-rational-minimax-fp64", 0.815),
                     ("single-batch-chebyshev-exact-division-fp64", 1.000),
                     ("single-batch-chebyshev-split-clenshaw-fp64", 1.200),
                     ("single-batch-rational-minimax-fast-fp64", 1.600))
SYNTHETIC_TIED = ("single-batch-chebyshev-fp64", "single-batch-rational-minimax-fp64")
SYNTHETIC_TRUTH = (f"the fastest in truth: {SYNTHETIC_ANCHOR}",
                   f"exactly tied with each other, 1.9% behind the fastest and inside this run's "
                   f"own resolution: {SYNTHETIC_TIED[0]} and {SYNTHETIC_TIED[1]}",
                   "25%, 50% and 100% behind the fastest, and outside that resolution: "
                   "single-batch-chebyshev-exact-division-fp64, "
                   "single-batch-chebyshev-split-clenshaw-fp64 and "
                   "single-batch-rational-minimax-fast-fp64",
                   "so the fit separates the last three, the tie set holds the first three, the "
                   "band is unresolved, and the pick is clause (2)'s and not the fit's fastest")


def emit_synthetic(path: pathlib.Path) -> int:
    """Write a small dump of the frozen format whose truth is known, so the analysis can be checked.

    The truth is the constants above: one class of six options, one of them genuinely 20.0% faster
    than the pack, two of them exactly tied, the rest at +15.0%, +40.0% and +80.0%, a per-round clock
    effect, a warm-up trend on the visit position, 2% log-normal noise on every visit, and the A/A
    column re-measuring the run's anchor once per round. The file says so in its own comments; the
    analysis is not told, and re-derives everything from the visits.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as bad:
        print(f"cannot write {path}: {bad.strerror or bad}")
        return 1

    generator = random.Random(SYNTHETIC_SEED)
    lines = ["# boys-option-probe raw dump v1",
             f"# protocol count={SYNTHETIC_COUNT} unit_target_ms={SYNTHETIC_UNIT_TARGET_MS:.2f} "
             f"passes={SYNTHETIC_PASSES} rounds={SYNTHETIC_ROUNDS} seed={SYNTHETIC_SEED} "
             f"reference={SYNTHETIC_ANCHOR} version=4.0.0",
             "# synthetic: written by tools/analyse_probe_dump.py --emit-synthetic",
             f"# synthetic truth: class fp64 single, {len(SYNTHETIC_OPTIONS)} options, "
             f"{SYNTHETIC_PASSES} passes x {SYNTHETIC_ROUNDS} rounds, seed {SYNTHETIC_SEED}",
             "# synthetic truth: the true nanoseconds per argument, and the same figure relative "
             "to the fastest,",
             "# synthetic truth: "
             + ", ".join(f"{name} {SYNTHETIC_BASE_NS * factor:.2f}ns "
                         f"({(factor / SYNTHETIC_OPTIONS[0][1] - 1.0) * 100.0:+.2f}%)"
                         for name, factor in SYNTHETIC_OPTIONS),
             f"# synthetic truth: 2% per-visit log-normal noise, a per-round clock effect and a "
             f"small warm-up trend on the visit position",
             f"# synthetic truth: the A/A column re-measures the anchor {SYNTHETIC_ANCHOR} once "
             f"per round"]

    for truth in SYNTHETIC_TRUTH:
        lines.append(f"# synthetic truth: {truth}")

    for name, factor in SYNTHETIC_OPTIONS:
        calibration = SYNTHETIC_BASE_NS * factor * math.exp(generator.gauss(0.0, SYNTHETIC_NOISE))
        repeats = max(1, round(SYNTHETIC_UNIT_TARGET_MS * 1e6 / (calibration * SYNTHETIC_COUNT)))
        lines.append(f"c {name} {calibration:.6g} {repeats}")

    timed = 0

    for pass_no in range(1, SYNTHETIC_PASSES + 1):
        for round_no in range(1, SYNTHETIC_ROUNDS + 1):
            round_effect = generator.gauss(0.0, SYNTHETIC_ROUND_NOISE)
            order = list(range(len(SYNTHETIC_OPTIONS)))
            generator.shuffle(order)
            visits: list[tuple[int, str, float]] = []

            for (name, factor), position in zip(SYNTHETIC_OPTIONS, order):
                true = math.log(SYNTHETIC_BASE_NS * factor) + round_effect \
                    + SYNTHETIC_POSITION_SLOPE * position
                visits.append((position, name,
                               math.exp(true + generator.gauss(0.0, SYNTHETIC_NOISE))))

            anchor_position = order[0]
            anchor_factor = SYNTHETIC_OPTIONS[0][1]
            true = math.log(SYNTHETIC_BASE_NS * anchor_factor) + round_effect \
                + SYNTHETIC_POSITION_SLOPE * anchor_position
            visits.append((len(SYNTHETIC_OPTIONS), f"{AA}{SYNTHETIC_ANCHOR}",
                           math.exp(true + generator.gauss(0.0, SYNTHETIC_NOISE))))

            for position, name, measured in sorted(visits):
                lines.append(f"t {pass_no} {round_no} {position} {measured:.6g} 1 {name}")
                timed += 1

    lines.append(f"# end lines={timed + len(SYNTHETIC_OPTIONS)}")

    try:
        path.write_text("\n".join(lines) + "\n", encoding="ascii")
    except OSError as bad:
        print(f"cannot write {path}: {bad.strerror or bad}")
        return 1

    print(f"synthetic dump written: {path}")
    print(f"  {len(SYNTHETIC_OPTIONS)} option(s) in one class (fp64 single), "
          f"{SYNTHETIC_PASSES} pass(es) x {SYNTHETIC_ROUNDS} round(s), "
          f"{len(SYNTHETIC_OPTIONS)} calibration line(s), {timed} timed visit(s)")
    print("  the truth it encodes, so the analysis can be checked against it:")
    print(f"    the true nanoseconds per argument, and the same figure relative to the fastest:")

    for name, factor in SYNTHETIC_OPTIONS:
        print(f"      {SYNTHETIC_BASE_NS * factor:>6.2f} ns  "
              f"{(factor / SYNTHETIC_OPTIONS[0][1] - 1.0) * 100.0:>+8.2f}%  {name}")

    for truth in SYNTHETIC_TRUTH:
        print(f"    {truth}")

    print(f"    2% per-visit log-normal noise, a per-round clock effect, a warm-up trend, seed "
          f"{SYNTHETIC_SEED}")
    print()

    return 0


# --- the command -------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Redo the option probe's ranking offline from its raw dump: the paired "
                    "log-time fit, the simultaneous intervals, the run's noise floor, its minimum "
                    "detectable effect, the tie set and the deterministic pick.")
    parser.add_argument("--dump", help="the raw dump to analyse")
    parser.add_argument("--emit-synthetic", metavar="PATH",
                        help="write a small synthetic dump of the frozen format whose truth is "
                             "known, then analyse it with --dump to verify this analysis with no "
                             "machine run")
    parser.add_argument("--min-class-options", type=int, default=2, metavar="N",
                        help="skip classes with fewer than N options, saying so (default 2)")
    parser.add_argument("--check", action="store_true",
                        help="the reporting contract, named: every class's tie set and pick are "
                             "printed in full however many options the tie set holds, and the exit "
                             "status is this analysis's own claim about completeness")
    args = parser.parse_args(argv)

    if args.emit_synthetic:
        status = emit_synthetic(pathlib.Path(args.emit_synthetic))

        if status != 0 or not args.dump:
            return status

    if not args.dump:
        parser.print_usage()
        print("analyse_probe_dump.py: --dump FILE or --emit-synthetic PATH is required")
        return 1

    if args.min_class_options < 1:
        print(f"analyse_probe_dump.py: --min-class-options takes a count of at least 1, not "
              f"{args.min_class_options}")
        return 1

    return analyse(pathlib.Path(args.dump), args.min_class_options, args.check)


def analyse(path: pathlib.Path, minimum: int, check: bool) -> int:
    """The whole report: read the dump, analyse every class, print it, and report completeness."""
    try:
        dump = parse_dump(path)
    except Unreadable as bad:
        print(f"unreadable: {bad}")
        return 1

    known = dump.format_version in KNOWN_FORMAT_VERSIONS
    emit_header(dump, known)

    if dump.malformed:
        print(f"  {len(dump.malformed)} line(s) this reader cannot place:")

        for number, raw in dump.malformed[:10]:
            print(f"    line {number}: {raw.strip()[:96]}")

        if len(dump.malformed) > 10:
            print(f"    and {len(dump.malformed) - 10} more")
        print()

    if not known:
        print(f"format version {dump.format_version} is not one this reader knows "
              f"({', '.join(str(version) for version in KNOWN_FORMAT_VERSIONS)}): a later version's "
              f"columns would be figures misread here, so nothing is analysed")
        return 1

    if not dump.visits:
        print("the dump carries no timed visit, so no class was analysed")
        return 1

    if check:
        print("check: the reporting contract - every class's tie set and pick are printed in full")
        print("however many options the tie set holds, and 'unresolved at +/- w' is printed where")
        print("the pick could not separate the tie. An unresolved class is an accepted outcome and")
        print("not a failure; the exit status is about completeness alone.")
        print()

    declared, conflicts = declared_classes(dump)
    emit_notes(dump, conflicts)

    reference = dump.protocol.get("reference", "")
    by_name = {visit.base_name for visit in dump.visits} | set(dump.calibrations)
    classes: dict[tuple[str, str], list[str]] = {}
    visits_by_class: dict[tuple[str, str], list[Visit]] = {}
    unplaced: list[str] = []

    for name in sorted(by_name):
        key = classify(name, declared)

        if key is None:
            unplaced.append(name)
        else:
            classes.setdefault(key, []).append(name)

    for visit in dump.visits:
        key = classify(visit.base_name, declared)

        if key is not None:
            visits_by_class.setdefault(key, []).append(visit)

    if reference and reference not in by_name:
        print(f"  the protocol line's reference={reference} is not a name this dump measures, so")
        print(f"  each class anchors on its own fastest option")
        print()

    print(f"  read      {len(dump.visits)} timed visit(s) over "
          f"{len({visit.round_key for visit in dump.visits})} round(s), {len(by_name)} distinct "
          f"option name(s), {len(classes)} class(es); {len(dump.calibrations)} calibration line(s)")

    if reference:
        print(f"  anchor    the protocol line names {reference} as the reference every ratio of "
              f"this run is formed against")
    print()

    ratios, ratio_notes = a_a_ratios(dump.visits, declared)
    class_ratios: dict[tuple[str, str], list[float]] = {}

    for _key, ratio, _name, placed in ratios:
        class_ratios.setdefault(placed, []).append(ratio)

    reports: list[ClassReport] = []
    panels: dict[tuple[str, str], RatioPanel] = {}
    skipped: dict[tuple[str, str], str] = {}

    for key in sorted(classes):
        report = analyse_class(key, classes[key], visits_by_class.get(key, []), reference, minimum,
                               dump.calibrations)
        reports.append(report)
        panels[key] = report.panel if report.panel is not None else RatioPanel(
            anchor="", anchor_from="", values=[], groups=[], rounds=[])

        if report.skipped:
            skipped[key] = report.skipped

    if ratio_notes:
        print("  the A/A column, as this reader placed it:")

        for note in ratio_notes:
            print(f"    {note}")
        print()

    emit_resolution(ratios, class_ratios, panels, skipped)

    print("PER CLASS")
    print()

    for report in reports:
        report_class(report, dump.calibrations)

    print("THE PICKS, ONE LINE PER CLASS")

    for report in reports:
        if report.skipped:
            print(f"  {display_class(report.key):<24} not analysed: {report.skipped}")
            continue

        unresolved = ""

        if len(report.tie) > 1:
            unresolved = (f", {len(report.tie)} tied of {len(report.options)}, unresolved at "
                          f"+/- {report.band:.2f}%")

        print(f"  {display_class(report.key):<24} {report.pick}   "
              f"({report.clause.split(':', 1)[0]}{unresolved})")

    print()

    analysed = [report for report in reports if not report.skipped]

    if unplaced:
        print(f"  {UNCLASSIFIED_NAME}: {len(unplaced)} option name(s) that neither the class rule")
        print(f"  nor a class column places, so no ranking here covers them:")

        for name in unplaced[:20]:
            print(f"    {name}")

        if len(unplaced) > 20:
            print(f"    and {len(unplaced) - 20} more")
        print()

    if dump.malformed:
        print(f"  {len(dump.malformed)} line(s) of the dump were not read: a ranking over part of a")
        print(f"  run is a ranking over the part that was read, and no more")
        print()

    if not analysed:
        print(f"verdict: no class could be analysed")
        print(f"         the exit status is this analysis's claim about the ranking, and there is "
              f"no ranking here")
        return 1

    if unplaced or dump.malformed or len(analysed) != len(reports):
        print(f"verdict: {len(analysed)} of {len(reports)} class(es) analysed")
        return 1

    print(f"verdict: every class was analysed ({len(reports)} of {len(reports)}), no option went")
    print(f"         unclassified, and every line of the dump was read")
    return 0


if __name__ == "__main__":
    sys.exit(main())
