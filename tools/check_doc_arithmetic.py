#!/usr/bin/env python3
"""Checks the arithmetic stated in the library's published prose.

The accuracy gate compares declared bounds against delivered values, so a
bound that has a verdict is a bound nobody has to re-derive. A *derived*
statement has no such check. "sixty times more accurate than the numbers used
to store it", "a margin of thousandths of a per cent", "five to eight orders
of magnitude past the 5.5e-14 bound" are arithmetic on numbers the same
sentence prints, and nothing in the repository recomputes them. Readers found
the errors in the calibration corpus below by doing exactly that arithmetic,
by hand.

This script does it mechanically, and it is deliberately noisy: a tool that
flags a candidate for a human to judge is worth more than one that silently
misses one. It never invents an operand. Where a claim's two sides are not
both printed, it says "not derivable" and names what is missing.

Three checks:

1. derived relations -- a sentence stating a relation between two or more
   numerals (times, x, factor of, orders of magnitude, per cent, parts in).
   The operands are recombined and the printed claim is compared with the
   result. Operands come from the claim's own sentence first, then its
   paragraph, then the document; the finding says which scope supplied each.

2. the accuracy multiplier m -- most documented budgets have the form m*B, and
   m defaults to 1. A claim that a cell "sits at 0.9999 of the bound" is a
   statement about m = 1 that holds only there. Every bound comparison is
   swept across the documented multiplier range and flagged when the number it
   prints moves with m and the passage does not say which setting it means.

3. one quantity, two values -- the same fraction or percentage printed two
   ways for the same named parameter (the same system size, order or
   argument). No single-sentence check can see this; it needs both sites.

Usage:
    python tools/check_doc_arithmetic.py                  # the published set
    python tools/check_doc_arithmetic.py docs/lane-contract.md
    python tools/check_doc_arithmetic.py --json
    python tools/check_doc_arithmetic.py --selftest       # calibration corpus

Exit status is 1 when a mismatch is found, and 0 otherwise. "Not derivable"
and "multiplier-unswept" are reports, not failures; --strict promotes the
latter to a failure, and --min-confidence governs how wide an operand scope a
mismatch may come from and still fail the run.
"""

import argparse
import json
import math
import pathlib
import re
import sys
from dataclasses import dataclass, field, replace

REPO = pathlib.Path(__file__).resolve().parent.parent

# The published set: what a reader outside this repository can open. The two
# documents under docs/ are here because they carry the per-lane contracts, so
# they make more quantitative claims than any other published file; a checker
# that skipped them would be checking the prose least likely to be wrong.
PUBLISHED = ("README.md", "CONTRIBUTING.md", "docs/lane-contract.md",
             "docs/consumer-perspective.md")

CALIBRATION_DOC = "tools/check_doc_arithmetic_calibration.md"

# The accuracy multiplier's documented range, in powers of two.
MULTIPLIERS = tuple(1 << k for k in range(17))

# How far a printed claim may sit from the recomputed value before it is a
# mismatch. A hedged claim ("about", "roughly", "approximately") gets the wider
# band; an unhedged one gets the narrow one. Deliberately tight: an "about"
# figure that is 10% out is exactly the class of error this checks for.
TOL_HEDGED = 0.05
TOL_PLAIN = 0.02

WORD_NUMBERS = {
    "half": 0.5, "one": 1.0, "two": 2.0, "twice": 2.0, "three": 3.0, "four": 4.0,
    "five": 5.0, "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0, "ten": 10.0,
    "eleven": 11.0, "twelve": 12.0, "thirteen": 13.0, "fourteen": 14.0,
    "fifteen": 15.0, "sixteen": 16.0, "twenty": 20.0, "thirty": 30.0, "forty": 40.0,
    "fifty": 50.0, "sixty": 60.0, "hundred": 100.0, "thousand": 1e3, "million": 1e6,
    "billion": 1e9, "trillion": 1e12,
}

WORD_FRACTIONS = {
    "tenth": 1e-1, "hundredth": 1e-2, "thousandth": 1e-3, "millionth": 1e-6,
}

# Kind hints. A kind is what makes two operands plausible as the two sides of
# one ratio -- an error bound and an error bound, not an error bound and a
# version number. Each hint carries a distance penalty and a weight on the
# distance to its right:
#
#   * a hint that ends within TIGHT_LINK characters of the numeral is its
#     unit outright ("the 5.5e-14 bound", "x = 16", "at order 32");
#   * otherwise only hints inside the numeral's own clause count, split at
#     commas and dashes, because "the relative error reaches 5.6e4, which is
#     five to eight orders past the 5.5e-14 bound" puts two magnitudes with
#     two different units in one sentence;
#   * an argument coordinate ("x = 28.98") sits near everything in a document
#     about a function of x, so it carries a penalty and counts double when it
#     trails.
#
# "orders of magnitude" is deliberately not a kind: it is the relation, and it
# is read as one by find_claims. Treating it as a unit would label the
# magnitude being compared rather than the comparison.
KIND_HINTS = (
    ("PERCENT", re.compile(r"per\s?cent|percent|%|parts?\s+in\b|share\s+of", re.I),
     0, 1),
    ("DIGIT", re.compile(r"ULP|representable\s+digit|significant\s+(?:bit|digit)"
                         r"|half-quantum|binary16|half\s+precision|bits\b", re.I),
     0, 1),
    ("ERROR", re.compile(r"bound|budget|error|accura|margin|precision"
                         r"|tolerance|rounding|residual", re.I), 0, 1),
    ("ARG", re.compile(r"\bx\s*[=≥>=]|argument|\border\b|\bn\b|boundary"
                       r"|region", re.I), 40, 2),
)
TIGHT_LINK = 6

# A unit defined in one section is a legitimate operand for a ratio stated in
# another -- "the half lane's bound allows for representing its result" is half
# a representable digit, defined a page above -- but only when the definition
# is tight enough that the numeral and its unit word are the same phrase.
TIGHT_UNIT_LINK = 12

# A step word names a single representable increment, as opposed to "bits",
# which names a format's width. Only a step word makes a numeral the unit side
# of a ratio stated in steps.
STEP_RE = re.compile(r"\bULP\b|representable\s+digit|half-quantum", re.I)
CLAUSE_BREAK_RE = re.compile(r"[,;:—–()\[\]]")

# What each relation compares, and so which operand kind it needs. None means
# the kind is resolved per claim from the claim's own wording.
RELATION_KINDS = {
    "ratio": None, "factor": None, "parts_in": None,
    "orders": "ERROR", "percent": None, "of_bound": "ERROR",
}

HEDGE_RE = re.compile(
    r"\b(about|roughly|approximately|nearly|around|close to|some)\b", re.I)
MARGIN_RE = re.compile(r"margin|headroom|spare|slack|cushion", re.I)
OCCUPANCY_RE = re.compile(r"of\s+the\s+whole\s+bound|of\s+the\s+bound"
                          r"|occupies|spends|consumes", re.I)

NUM_RE = re.compile(
    r"(?<![\w.])"
    r"(?P<mant>\d+(?:\.\d+)?)"
    r"(?:"
    r"(?:\s*[eE]\s*(?P<exp>[+-]?\d+))"
    r"|(?:\s*[*·×]\s*10\s*\^?\s*(?P<pow>[+-]?\d+))"
    r"|(?:\s*\^\s*(?P<caret>[+-]?\d+))"
    r")?"
)

WORD_RE = re.compile(
    r"\b(" + "|".join(sorted(WORD_NUMBERS, key=len, reverse=True)) + r")\b", re.I)
FRACTION_WORD_RE = re.compile(
    r"\b(" + "|".join(sorted(WORD_FRACTIONS, key=len, reverse=True)) + r")s?\b", re.I)
PARTS_IN_RE = re.compile(r"\s+parts?\s+in\s+(?:a\s+|one\s+|an?\s+)?", re.I)
NUMBER_WORD_RE = re.compile(
    r"\b(?:" + "|".join(sorted(WORD_NUMBERS, key=len, reverse=True)) + r")\b", re.I)


def read_number_words(text, pos):
    """Reads a run of number words as one value: "ten thousand" is 10000.

    Reading the words one at a time gives 10 and 1000, and a ratio taken
    against the first of those is a wrong number presented as a checked one.
    Returns (value, end), or (None, pos) when nothing starts at `pos`.
    """
    total = 0.0
    current = 0.0
    end = pos
    seen = False
    for match in NUMBER_WORD_RE.finditer(text, pos):
        if text[end:match.start()].strip():     # only whitespace may separate
            break
        value = WORD_NUMBERS[match.group(0).lower()]
        if value >= 100:
            current = (current or 1.0) * value
            if value >= 1000:
                total += current
                current = 0.0
        else:
            current += value
        end = match.end()
        seen = True
    if not seen:
        return None, pos
    return total + current, end
# The named parameter a number is a value of. Two shapes: the name then the
# number ("order 3", "n = 5", "basis functions 586") and the number then the
# unit ("586 BF", "586-basis-function").
KEY_NUMBER_RE = re.compile(
    r"(?:"
    r"\b(?:order|n|size|system\s+size|count|atom\s+count)\s*[=:]?\s*"
    r"(?P<v1>\d+(?:\.\d+)?)"
    r"|\b(?:basis\s+functions?|BF)\s*[=:]?\s*(?P<v2>\d+(?:\.\d+)?)"
    r"|(?P<v3>\d+(?:\.\d+)?)[\s-]*(?:BF\b|basis[\s-]+functions?)"
    r")", re.I)

# "\b" belongs on the word alternatives only: "%" is not a word character, so
# a trailing \b after it never matches. Getting this wrong silently drops every
# numeric percentage in the corpus.
PERCENT_TAIL_RE = re.compile(r"\s*(?:%|per\s?cent\b|percent\b)", re.I)

# "8.9 % at 586" is a fraction carrying its own parameter, and a sentence can
# carry a whole ladder of them ("71 % at 34 BF, 38 % at 106, ... 8.9 % at 586").
# Each percentage takes the parameter it is printed against, not the sentence's
# longest match: attributing the ladder to one key compares five sizes.
AT_KEY_RE = re.compile(r"\bat\s+(\d+(?:\.\d+)?)")

# A list item is its own passage. List items sit on consecutive lines with no
# blank line between them, so joining them makes one sentence out of a whole
# bullet list and attributes every number in it to one subject.
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s)")

SEVERITY = {"mismatch": 0, "multiplier-unswept": 1, "not derivable": 2,
            "verified": 3}
SCOPE_DEPTH = {"sentence": 0, "paragraph": 1, "section": 2, "unit definition": 2,
               "document": 2}


# --------------------------------------------------------------------------
# Reading the documents
# --------------------------------------------------------------------------

@dataclass
class Line:
    number: int
    text: str
    kind: str               # text | table | fence | generated | other


@dataclass
class Numeral:
    start: int
    end: int
    text: str
    value: float
    kind: str = "PLAIN"
    scope: str = "sentence"
    in_code: bool = False
    scaled: bool = False
    derived: str = ""
    tight_unit: bool = False    # its unit word is in its own phrase
    fraction_word: bool = False  # "thousandths": a fraction, not a count
    step_unit: bool = False     # its unit word is a representable increment

    def show(self):
        return f"{self.derived or self.text} [{self.kind}]"


@dataclass
class Claim:
    start: int
    end: int
    text: str
    relation: str
    claimed: float
    claimed_hi: float = None
    hedged: bool = False
    needed_kind: str = None
    sentence: object = None
    hi_end: int = None          # end of a "five to eight" range, so the upper
                                # numeral is not also read as an operand

    def span(self):
        if self.claimed_hi is not None and self.claimed_hi != self.claimed:
            return f"{self.claimed:g} to {self.claimed_hi:g}"
        return f"{self.claimed:g}"

    def holds(self, value, tolerance):
        lo = self.claimed
        hi = self.claimed_hi if self.claimed_hi is not None else self.claimed
        return lo * (1 - tolerance) <= value <= hi * (1 + tolerance)


@dataclass
class Finding:
    check: str
    verdict: str
    path: str
    line: int
    claim: str
    sentence: str
    operands: list = field(default_factory=list)
    arithmetic: str = ""
    detail: str = ""

    def as_dict(self):
        return {"check": self.check, "verdict": self.verdict, "path": self.path,
                "line": self.line, "claim": self.claim, "sentence": self.sentence,
                "operands": [{"text": t, "value": v, "kind": k, "scope": s}
                             for (t, v, k, s) in self.operands],
                "arithmetic": self.arithmetic, "detail": self.detail}

    def worst_scope(self):
        return max((SCOPE_DEPTH.get(s, 2) for (_, _, _, s) in self.operands),
                   default=2)


@dataclass
class Sentence:
    id: str
    line: int
    text: str
    start: int
    numerals: list = field(default_factory=list)

    def slice(self, a, b):
        return self.text[max(0, a - self.start):max(0, b - self.start)]


@dataclass
class Paragraph:
    id: str
    line: int
    text: str
    sentences: list = field(default_factory=list)
    section: int = 0
    start: int = 0          # offset of this paragraph in the document's flat

    @property
    def numerals(self):
        return [n for s in self.sentences for n in s.numerals]


@dataclass
class Doc:
    path: str
    lines: list
    flat: str
    sentences: list
    paragraphs: list
    numerals: list
    line_starts: dict

    def section_numerals(self, para):
        """Every numeral between the same two headings as `para`."""
        if para is None:
            return []
        return [n for other in self.paragraphs if other.section == para.section
                for n in other.numerals]

    def tight_units(self, kind):
        """Numerals the document attaches to a unit word in their own phrase.

        A unit defined in one section is the operand of a ratio stated in
        another, but only where the definition is a phrase and not a clause
        away: "half a representable digit" defines the unit, "worst measured
        4.243, for arguments at or above x = 28.984375" does not.
        """
        return [n for n in self.numerals
                if not n.in_code and n.kind == kind and n.tight_unit]


def read_lines(path):
    """Splits a document into classified lines.

    A generated block and a fenced code block are not the library's own
    sentences, so they are excluded from claim detection. They keep their
    offsets so that line numbers stay honest.
    """
    raw = path.read_text(encoding="utf-8").splitlines()
    out = []
    in_fence = False
    in_generated = False
    for index, text in enumerate(raw, start=1):
        stripped = text.strip()
        if "<!-- platform-table:begin" in text:
            in_generated = True
        if in_generated:
            out.append(Line(index, text, "generated"))
            if "<!-- platform-table:end" in text:
                in_generated = False
            continue
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(Line(index, text, "fence"))
        elif in_fence:
            out.append(Line(index, text, "fence"))
        elif stripped.startswith("<!--"):
            out.append(Line(index, text, "other"))
        elif stripped.startswith("|"):
            out.append(Line(index, text, "table"))
        else:
            out.append(Line(index, text, "text"))
    return out


def code_spans(text):
    """The [start, end) ranges covered by backtick code spans on one line."""
    spans = []
    start = None
    for i, ch in enumerate(text):
        if ch == "`":
            if start is None:
                start = i
            else:
                spans.append((start, i))
                start = None
    if start is not None:
        spans.append((start, len(text)))
    return spans


def split_sentences(body, base):
    """Cuts a passage into sentences, returning absolute offsets."""
    out = []
    start = 0
    # "/" starts a sentence so that a header's Doxygen prose, whose every line
    # begins with "///", does not collapse into one passage per doc comment.
    for match in re.finditer(r"(?<=[.!?])\s+(?=[A-Z*“\"(/])", body):
        chunk = body[start:match.start()]
        if chunk.strip():
            out.append((base + start, chunk))
        start = match.end()
    tail = body[start:]
    if tail.strip():
        out.append((base + start, tail))
    return out


def _hint_distance(hit, start, end, trailing_weight):
    if hit.end() <= start:
        return start - hit.end()
    if hit.start() >= end:
        return (hit.start() - end) * trailing_weight
    return 0


def _best_hint(text, start, end, region=None):
    """The best-scoring hint, optionally restricted to one region of text."""
    best = None
    for rank, (kind, pattern, penalty, weight) in enumerate(KIND_HINTS):
        for hit in pattern.finditer(text):
            if region is not None and not (region[0] <= hit.start()
                                           and hit.end() <= region[1]):
                continue
            distance = _hint_distance(hit, start, end, weight)
            score = distance + penalty
            if best is None or (score, rank) < (best[0], best[1]):
                best = (score, rank, kind)
    return best


def _score_kinds(text, start, end):
    """The kind of the numeral occupying [start, end) in `text`."""
    tight = _best_hint(text, start, end)
    if tight is not None and tight[0] <= TIGHT_LINK:
        return tight[2]

    left, right = 0, len(text)
    for hit in CLAUSE_BREAK_RE.finditer(text):
        if hit.end() <= start:
            left = hit.end()
        elif hit.start() >= end:
            right = hit.start()
            break
    clause = _best_hint(text, start, end, (left, right))
    if clause is not None:
        return clause[2]

    whole = _best_hint(text, start, end)
    return whole[2] if whole else "PLAIN"


def extract_numerals(line):
    """Every numeral on one line, with its value, kind and flags."""
    spans = code_spans(line.text)
    found = []

    def in_code(pos):
        return any(a <= pos < b for (a, b) in spans)

    for match in NUM_RE.finditer(line.text):
        mant = float(match.group("mant"))
        try:
            if match.group("exp") is not None:
                value = mant * 10.0 ** int(match.group("exp"))
            elif match.group("pow") is not None:
                value = mant * 10.0 ** int(match.group("pow"))
            elif match.group("caret") is not None:
                value = mant ** int(match.group("caret"))
            else:
                value = mant
        except (ValueError, OverflowError):
            continue
        found.append(Numeral(
            start=match.start(), end=match.end(),
            text=line.text[match.start():match.end()], value=value,
            in_code=in_code(match.start()),
            scaled=bool(re.search(r"\bm\s*[*·×]\s*$",
                                  line.text[:match.start()]))))

    for match in WORD_RE.finditer(line.text):
        if in_code(match.start()):
            continue
        word = match.group(1).lower()
        found.append(Numeral(
            start=match.start(), end=match.end(), text=match.group(1),
            value=WORD_NUMBERS[word], derived=f"word '{word}'"))

    for match in FRACTION_WORD_RE.finditer(line.text):
        if in_code(match.start()):
            continue
        word = match.group(1).lower()
        if word not in WORD_FRACTIONS:
            continue
        found.append(Numeral(
            start=match.start(), end=match.end(), text=match.group(0),
            value=WORD_FRACTIONS[word], kind="PERCENT", fraction_word=True,
            derived=f"word '{word}'"))

    deduped = []
    for num in sorted(found, key=lambda n: (n.start, n.end)):
        if any(num.start < other.end and other.start < num.end for other in deduped):
            continue
        deduped.append(num)
    return deduped


def score_numerals(text, base, numerals):
    """Gives each numeral its kind and tight-unit flag against `text`.

    Scoring runs over the paragraph, not the line: a unit word wraps ("...half
    of one
    representable digit...") and a line-scoped lookup would lose it.
    """
    for num in numerals:
        start, end = num.start - base, num.end - base
        if num.fraction_word:
            continue                    # "thousandths" is a fraction by itself
        num.kind = _score_kinds(text, start, end)
        patterns = [p for (k, p, _, _) in KIND_HINTS if k == num.kind]
        num.tight_unit = any(
            _hint_distance(hit, start, end, 1) <= TIGHT_UNIT_LINK
            for pattern in patterns for hit in pattern.finditer(text))
        num.step_unit = any(
            _hint_distance(hit, start, end, 1) <= TIGHT_UNIT_LINK
            for hit in STEP_RE.finditer(text))


def build_doc(path):
    """Parses one document into paragraphs, sentences and numerals.

    Offsets are absolute in `flat`, which is the document with the excluded
    line kinds blanked out but still occupying their newlines, so every offset
    maps back to a real line number.
    """
    lines = read_lines(path)
    line_start = {}
    chunks = []
    cursor = 0
    for line in lines:
        line_start[line.number] = cursor
        body = line.text if line.kind == "text" else ""
        chunks.append(body)
        cursor += len(body) + 1
    flat = "\n".join(chunks)

    numerals_by_line = {}
    for line in lines:
        if line.kind in ("text", "table"):
            numerals_by_line[line.number] = [
                replace(n, start=n.start + line_start[line.number],
                        end=n.end + line_start[line.number])
                for n in extract_numerals(line)]

    by_offset = {}
    for line in lines:
        for num in numerals_by_line.get(line.number, []):
            by_offset[num.start] = num

    paragraphs = []
    sentences = []
    section = 0
    buf = []

    def flush(block, first_line, section_id):
        start = line_start[first_line]
        end = line_start[block[-1].number] + len(block[-1].text)
        body = flat[start:end]
        para = Paragraph(f"{path.name}:{first_line}", first_line, body,
                         section=section_id, start=start)
        for (offset, text) in split_sentences(body, start):
            para.sentences.append(Sentence(f"{para.id}@{offset}", first_line, text,
                                           offset))
        paragraphs.append(para)
        sentences.extend(para.sentences)

    for line in lines:
        if line.kind == "text" and line.text.strip():
            heading = line.text.lstrip().startswith("#")
            item = bool(LIST_ITEM_RE.match(line.text))
            if (heading or item) and buf:
                flush(buf, buf[0].number, section)
                buf = []
            if heading:
                section = line.number
            buf.append(line)
            continue
        if buf:
            flush(buf, buf[0].number, section)
            buf = []
        if line.kind == "table" and line.text.strip():
            flush([line], line.number, section)
    if buf:
        flush(buf, buf[0].number, section)

    for sent in sentences:
        for num in by_offset.values():
            if sent.start <= num.start < sent.start + len(sent.text):
                sent.numerals.append(num)
        sent.numerals.sort(key=lambda n: n.start)

    for para in paragraphs:
        score_numerals(para.text, para.start, para.numerals)

    all_numerals = sorted(by_offset.values(), key=lambda n: n.start)
    for num in all_numerals:
        num.scope = "document"

    doc = Doc(str(path), lines, flat, sentences, paragraphs, all_numerals,
              line_start)
    return doc


def line_of(doc, offset):
    best = doc.lines[0].number if doc.lines else 1
    for number, start in doc.line_starts.items():
        if start <= offset:
            best = number
    return best


def paragraph_of(doc, sentence):
    for para in doc.paragraphs:
        if sentence in para.sentences:
            return para
    return None


# --------------------------------------------------------------------------
# Check 1 -- derived relations
# --------------------------------------------------------------------------

def find_claims(doc):
    """Every place the prose states a relation over numbers it also prints."""
    claims = []
    for sent in doc.sentences:
        covered_until = 0
        for num in sent.numerals:
            if num.in_code or num.scaled or num.end <= covered_until:
                continue
            after = sent.slice(num.end, num.end + 90)
            before = sent.slice(num.start - 45, num.start)
            relation = None
            claimed = num.value
            claimed_hi = None
            span_end = num.end

            # As with "%", the multiplication sign is not a word character, so
            # "\b" goes on the spelled-out alternatives only.
            if re.match(r"\s*(?:times\b|time\b|-fold\b|×)", after, re.I):
                relation = "ratio"

            if relation is None and re.match(
                    r"\s*(?:to\s+(?:[a-z]+|\d+(?:\.\d+)?)\s+)?orders?\s+of\s+magnitude",
                    after, re.I):
                relation = "orders"
                span = re.match(r"\s*to\s+([a-z]+|\d+(?:\.\d+)?)", after, re.I)
                if span:
                    # "five to eight orders" is one claim, not two: the upper
                    # numeral is part of this one's span.
                    span_end = num.end + span.end()
                    token = span.group(1).lower()
                    claimed_hi = WORD_NUMBERS.get(token)
                    if claimed_hi is None:
                        try:
                            claimed_hi = float(token)
                        except ValueError:
                            claimed_hi = None

            if (relation is None and num.kind == "PERCENT"
                    and num.derived and re.match(r"\s+of\s+a\s+per\s?cent", after,
                                                 re.I)):
                relation = "percent"

            if relation is None and PERCENT_TAIL_RE.match(after):
                relation = "percent"

            if relation is None and re.search(r"\bfactor\s+of\s*$", before, re.I):
                relation = "factor"

            if relation is None and re.match(r"\s*of\s+the\s+(?:bound|budget)\b",
                                             after, re.I) and 0 < num.value <= 1.0:
                relation = "of_bound"

            if relation is None:
                hit = PARTS_IN_RE.match(after)
                if hit:
                    denom, _ = read_number_words(after, hit.end())
                    if denom is None:
                        token = re.match(r"\d+(?:\.\d+)?", after[hit.end():])
                        denom = float(token.group(0)) if token else None
                    if denom:
                        relation = "parts_in"
                        claimed = num.value / denom

            if relation is None:
                continue

            hedged = bool(HEDGE_RE.search(before[-26:])) or bool(
                HEDGE_RE.search(after[:18]))
            claim = Claim(start=num.start, end=num.end, text=num.text,
                          relation=relation, claimed=claimed, claimed_hi=claimed_hi,
                          hedged=hedged, sentence=sent)
            if relation == "ratio":
                claim.needed_kind = kind_at(sent.text, num.end - sent.start + 1)
                # "54 times tighter" compares two accuracies whatever noun the
                # sentence's nearest kind word happens to be.
                if claim.needed_kind in ("PLAIN", "ARG") and re.search(
                        r"accura|error|bound|budget|tighter|looser|better|worse",
                        after, re.I):
                    claim.needed_kind = "ERROR"
                # A sentence that measures in representable digits or
                # significant bits is comparing those, whatever noun the claim's
                # own wording happens to lean on: "loses 7.7 times as much as
                # the half lane's bound allows" is a ratio of representation
                # steps, not of two error magnitudes.
                if claim.needed_kind != "PERCENT" and any(
                        n.kind == "DIGIT" for n in sent.numerals):
                    claim.needed_kind = "DIGIT"
            elif relation == "orders":
                claim.needed_kind = "ERROR"
                if claimed_hi is not None:
                    claim.hi_end = span_end
            elif relation == "percent":
                claim.needed_kind = None          # percent decides its own operand
            elif relation == "factor":
                claim.needed_kind = "PLAIN"
            elif relation == "of_bound":
                claim.needed_kind = "ERROR"
            claims.append(claim)
            covered_until = max(covered_until, span_end)
    return claims


def kind_at(text, index):
    """The kind hint around an index into `text`."""
    return _score_kinds(text, index, index)


def operand_pool(doc, claim, needed_kind):
    """The operands for a claim, widest scope last, provenance recorded.

    Stops at the first scope that supplies two operands of the required kind.
    Returns (pool, scope, kinds_present). Nothing is invented: a scope that
    cannot supply two operands of the right kind is reported as such.
    """
    sent = claim.sentence
    para = paragraph_of(doc, sent) if sent else None
    # Each entry is (scope name, [(numeral, provenance)]). Provenance travels
    # with the operand rather than on it: the same numeral is a sentence
    # operand of one claim and a section operand of another, and tagging it in
    # place makes the second reading inherit the first.
    ladder = [
        ("sentence", [(n, "sentence") for n in (sent.numerals if sent else [])]),
        ("paragraph", [(n, "paragraph") for n in (para.numerals if para else [])]),
    ]
    # A cross-section sweep of error magnitudes is how the document would name
    # every bound at once, so an ERROR-kind claim stops at its paragraph. What
    # is worth taking is DIGIT, and in two pieces: the claim's own section,
    # whose other paragraphs hold the lane's bound and its measured worst
    # ("within 8 ... worst measured 4.243"), and the unit words the document
    # defines tightly elsewhere ("half a representable digit"). A unit defined
    # a section away is a legitimate operand; a bare number a section away is
    # not, and pairing those is how a checker finds ratios that mean nothing.
    if needed_kind in (None, "PLAIN", "DIGIT", "PERCENT"):
        section = [(n, "section") for n in doc.section_numerals(para)]
        if needed_kind in ("DIGIT", "PERCENT"):
            known = {id(n) for (n, _) in section}
            section += [(n, "unit definition")
                        for n in doc.tight_units(needed_kind) if id(n) not in known]
        ladder.append(("section", section))

    for scope, candidates in ladder:
        # A numeral inside a multiplier expression ("m*5.5e-14") is a usable
        # operand at its printed value: the multiplier defaults to 1, so that
        # is what the expression equals, and check 2 sweeps the setting
        # separately. 0 and 1 are dropped -- they carry no magnitude.
        pool = [(n, where) for (n, where) in candidates
                if not n.in_code and n.value not in (0.0, 1.0)
                and not (n.start == claim.start and n.end == claim.end)
                and not (claim.hi_end is not None and claim.start <= n.start
                         and n.end <= claim.hi_end)
                and (needed_kind in (None, "PLAIN") or n.kind == needed_kind)]
        if len(pool) >= 2:
            return _dedupe(pool), scope, None

    present = {}
    for num in (sent.numerals if sent else []):
        if not num.in_code and not (
                num.start == claim.start and num.end == claim.end):
            present[num.kind] = present.get(num.kind, 0) + 1
    for num in (para.numerals if para else []):
        if not num.in_code:
            present.setdefault(num.kind, 0)
    reached = [name for (name, _) in ladder]
    return [], "none", (present, reached)


def _dedupe(nums):
    """Drops repeated operands: the same value read twice is one operand."""
    seen = {}
    for (num, where) in nums:
        key = (round(num.value, 12), num.kind)
        if key not in seen:
            seen[key] = (num, where)
    return list(seen.values())


def ratio_pairs(pool, require_step=False):
    """Every ordered pair whose ratio is at least one.

    When the claim is stated in representable digits, one side of the ratio is
    a single step of that unit and the other is a count of it. Requiring one of
    the two operands to be a step -- smaller than one, or attached to a step
    word of its own -- keeps a checker from pairing two unrelated numbers that
    happen to sit near the printed ratio. 4.243 over 0.5 is 8.5, and 0.5 is
    what the half lane's representation allowance is; 4.243 over a format width
    is a coincidence, and "32-bit" is a width however tight its "bit" reads.
    """
    out = []
    for i, (a, where_a) in enumerate(pool):
        for j, (b, where_b) in enumerate(pool):
            if i == j or b.value == 0:
                continue
            if require_step and not (a.value < 1.0 or b.value < 1.0
                                     or a.step_unit or b.step_unit):
                continue
            ratio = a.value / b.value
            if ratio >= 1.0:
                out.append((ratio, a, b, where_a, where_b))
    return out


def check_relations(doc):
    findings = []
    for claim in find_claims(doc):
        sent = claim.sentence
        finding = Finding("relation", "", doc.path, line_of(doc, claim.start),
                          claim.span(), sent.text if sent else "")

        if claim.relation == "percent":
            finding.verdict, finding.operands, finding.arithmetic, finding.detail = \
                percent_verdict(doc, claim)
            findings.append(finding)
            continue

        if claim.relation == "of_bound":
            continue                        # the input to check 2

        pool, scope, present = operand_pool(doc, claim, claim.needed_kind)
        if not pool:
            finding.verdict = "not derivable"
            counts, reached = present if present else ({}, [])
            where = " and ".join(reached) if reached else "the passage"
            if claim.needed_kind and claim.needed_kind != "PLAIN":
                finding.detail = (
                    f"the claim compares two {claim.needed_kind} magnitudes, and there "
                    f"are not two of them to compare: the sentence prints "
                    f"{counts.get(claim.needed_kind, 0)} of kind {claim.needed_kind} "
                    f"and no other {claim.needed_kind} follows in its paragraph. One "
                    f"side of the relation has no printed value, so no ratio is "
                    f"computable from the document, and inventing one would be worse "
                    f"than reporting. (Scopes searched: {where}. The whole document is "
                    f"not swept for this kind: pairing every bound in the library with "
                    f"every other would answer a different question.)")
            else:
                finding.detail = (f"fewer than two usable operands in {where}.")
            findings.append(finding)
            continue

        finding.operands = [(o.derived or o.text, o.value, o.kind, where)
                            for (o, where) in pool]

        if claim.relation in ("ratio", "factor"):
            pairs = ratio_pairs(pool, require_step=(claim.needed_kind == "DIGIT"))
            if not pairs:
                finding.verdict = "not derivable"
                finding.detail = "no operand pair gives a ratio of one or more."
                findings.append(finding)
                continue
            tolerance = TOL_HEDGED if claim.hedged else TOL_PLAIN
            ranked = sorted(pairs, key=lambda p: abs(p[0] / claim.claimed - 1.0))
            best = ranked[0]
            away = abs(best[0] / claim.claimed - 1.0)
            finding.arithmetic = (
                f"{best[1].show()} / {best[2].show()} = {best[0]:.6g}, against the "
                f"claimed {claim.claimed:g} ({away * 100:.1f}% away)")
            if away <= tolerance:
                finding.verdict = "verified"
            else:
                finding.verdict = "mismatch"
                plausible = [p for p in ranked
                             if max(p[0], claim.claimed) / min(p[0], claim.claimed) <= 4]
                finding.detail = ("candidate ratios from the printed operands: "
                                  + "; ".join(f"{p[1].derived or p[1].text}/"
                                              f"{p[2].derived or p[2].text} = {p[0]:.4g}"
                                              for p in plausible[:6])
                                  + f". Claimed {claim.claimed:g}, tolerance "
                                    f"{tolerance * 100:.0f}% "
                                    f"({'hedged' if claim.hedged else 'unhedged'}). "
                                    f"Operands came from the {scope}.")
            findings.append(finding)

        elif claim.relation == "orders":
            pairs = ratio_pairs(pool)
            scored = [(math.log10(r), a, b) for (r, a, b, _, _) in pairs if r > 1.0]
            if not scored:
                finding.verdict = "not derivable"
                finding.detail = "no operand pair gives a ratio greater than one."
                findings.append(finding)
                continue
            scored.sort(key=lambda s: -s[0])
            kept = []
            for (orders, a, b) in scored:
                entry = (round(orders, 4), a, b)
                if entry[0] not in [k[0] for k in kept]:
                    kept.append(entry)
            hi = claim.claimed_hi if claim.claimed_hi is not None else claim.claimed
            finding.arithmetic = "; ".join(
                f"log10({a.show()}/{b.show()}) = {o:.2f} orders"
                for (o, a, b) in kept[:3])
            if any(claim.holds(o, TOL_HEDGED if claim.hedged else TOL_PLAIN)
                   for (o, _, _) in kept):
                finding.verdict = "verified"
            else:
                finding.verdict = "mismatch"
                nearest = kept[0]
                finding.detail = (
                    f"the two printed magnitudes are {nearest[0]:.2f} orders apart "
                    f"({nearest[1].derived or nearest[1].text} against "
                    f"{nearest[2].derived or nearest[2].text}), against a stated "
                    f"{claim.claimed:g} to {hi:g}. Operands came from the {scope}.")
            findings.append(finding)

        elif claim.relation == "parts_in":
            finding.verdict = "verified"
            finding.arithmetic = (
                f"the claim prints both operands, so it recomputes exactly: "
                f"{claim.text} parts in the stated denominator = {claim.claimed:g}")
            findings.append(finding)

    return findings


def percent_verdict(doc, claim):
    """Recomputes a percentage claim, or says why it cannot be.

    Two shapes are distinguished, because they derive differently:

      margin    -- "0.9999 of the bound -- a margin of thousandths of a per
                   cent". The claim states a ratio and a headroom derived from
                   it: margin = 100 * (1 - ratio).
      occupancy -- "99% of the whole bound". The claim states what share of a
                   quantity one term takes. Its operand is that term, which
                   this document prints as a description ("half of one
                   representable digit"), not as a number.
    """
    sent = claim.sentence
    para = paragraph_of(doc, sent)
    window = sent.slice(claim.start - 60, claim.end + 40)
    is_margin = bool(MARGIN_RE.search(window))
    is_occupancy = bool(OCCUPANCY_RE.search(sent.slice(claim.end, claim.end + 40))
                        or OCCUPANCY_RE.search(window))

    if is_occupancy and not is_margin:
        return ("not derivable", [],
                "",
                "the claim states what share of a quantity one term takes, and the "
                "term is printed as a description ('half of one representable "
                "digit'), not as a number. No ratio is computable from the sentence.")

    ladder = (("sentence", sent.numerals if sent else []),
              ("paragraph", para.numerals if para else []))
    for scope, candidates in ladder:
        ratios = [n for n in candidates
                  if not n.in_code and 0 < n.value <= 1.0
                  and not (n.start == claim.start and n.end == claim.end)]
        if ratios:
            ratio = max(ratios, key=lambda n: n.value)
            computed = 100.0 * (1.0 - ratio.value)
            tolerance = TOL_HEDGED if claim.hedged else TOL_PLAIN
            operands = [(ratio.derived or ratio.text, ratio.value, ratio.kind, scope)]
            ratio_of = computed / claim.claimed if claim.claimed else float("inf")
            arithmetic = (f"margin = 100 * (1 - {ratio.show()}) = {computed:.6g} per "
                          f"cent, against the claimed {claim.claimed:g} per cent "
                          f"(a factor of {ratio_of:.2f})")
            if abs(ratio_of - 1.0) <= tolerance:
                return ("verified", operands, arithmetic, "")
            return ("mismatch", operands, arithmetic,
                    f"the printed cell is {ratio.value:g} of the bound, so the margin "
                    f"it leaves is {computed:.4g} per cent, not {claim.claimed:g} per "
                    f"cent. Operands came from the {scope}.")

    return ("not derivable", [], "",
            "no ratio in the sentence or its paragraph to derive a margin from, so "
            "the percentage stands alone with no second printed operand.")


# --------------------------------------------------------------------------
# Check 2 -- the multiplier m
# --------------------------------------------------------------------------

def check_multiplier(doc):
    """Sweeps every bound comparison over the documented multiplier range.

    A budget written m*B is a budget of B at the default m = 1. A sentence
    saying a cell sits at 0.9999 of the bound is a statement about m = 1, and
    it is a different number at every larger m. The multiplier's documented
    range runs to 65536, so the setting is spendable and the sentence has to
    name it or be swept.
    """
    findings = []
    for claim in find_claims(doc):
        if claim.relation != "of_bound":
            continue
        sent = claim.sentence
        para = paragraph_of(doc, sent)
        context = (sent.text if sent else "") + "\n" + (para.text if para else "")
        if re.search(r"at\s+m\s*=\s*1\b|default\s+multiplier|multiplier\s+of\s+one"
                     r"|at\s+the\s+default", context, re.I):
            continue
        swept = claim.claimed
        if not 0 < swept <= 1.0:
            continue
        # The lane's own paragraph may declare itself multiplier-free: the
        # native half lane does, and the region-C form does.
        if re.search(r"no\s+multiplier|multiplier\s+relaxes|no\s+m\b", context, re.I):
            continue
        # Whether the bound this compares against carries the multiplier is a
        # fact about the lane's section, and the section's bound expression may
        # sit a paragraph above the claim. Look for any m*B in the document;
        # none means there is nothing to sweep.
        finding = Finding("multiplier", "multiplier-unswept", doc.path,
                          line_of(doc, claim.start), claim.span(),
                          sent.text if sent else "")
        finding.operands = [(claim.text, swept, "of_bound", "sentence")]
        whole = doc.flat
        product_bound = bool(re.search(
            r"m\s*[*·×]\s*\d[\d.]*\s*(?:[eE][+-]?\d+)?", whole, re.I))
        sum_bound = bool(re.search(
            r"m\s*[*·×]\s*\d[\d.]*\s*(?:[eE][+-]?\d+)?[\s`*]*(?:plus|\+)",
            whole, re.I))
        if sum_bound:
            finding.arithmetic = (
                f"at m = 1: {swept:g} of the bound. The bound this document states "
                f"for the passage is a sum, m*B + c, with c printed as a description "
                f"('half of the last representable digit') and not as a number, so "
                f"the ratio at m = 2 is not computable from the document.")
            finding.detail = (
                "the passage's bound has the form m*B plus a fixed term, so the "
                f"printed {swept:g} is a value at one multiplier. Raising m loosens "
                "only the m*B part, so the ratio moves with m and the sentence does "
                "not say at which setting it was taken or what the fixed term is. "
                "Either the setting or the term has to be printed for the claim to "
                "be checkable.")
        elif product_bound:
            ladder = [(m, swept / m) for m in MULTIPLIERS]
            finding.arithmetic = "; ".join(
                f"m={m}: {swept:g}/{m} = {r:.6g}" for (m, r) in ladder[:6]) + " ..."
            over = [m for (m, r) in ladder if r >= 1.0]
            if over:
                finding.detail = (
                    f"the claim is a ratio to a budget the documents scale by m, and "
                    f"it is stated at m = 1. It reaches 1.0 of the budget at m = "
                    f"{min(over)}, so it is false at every multiplier from "
                    f"{min(over)} to {MULTIPLIERS[-1]}. The passage does not name the "
                    f"setting.")
            else:
                finding.detail = (
                    f"the claim is a ratio to a budget the documents scale by m, and "
                    f"it is stated at m = 1: {swept:g} holds only there. At m = 2 it "
                    f"is {swept / 2:.6g}. The verdict does not flip, but the printed "
                    f"number does, and the passage does not name the setting.")
        else:
            continue
        findings.append(finding)
    return findings


# --------------------------------------------------------------------------
# Check 3 -- one quantity, two values
# --------------------------------------------------------------------------

def check_agreement(docs):
    """The same fraction for the same named parameter, printed two ways.

    Only percentage and fraction claims are grouped. Every other number is
    left alone: two different arguments legitimately carry two different
    bounds, and grouping them by their nearest noun would drown the check in
    those.
    """
    groups = {}
    for doc in docs:
        for sent in doc.sentences:
            keys = [(match.group("v1") or match.group("v2") or match.group("v3"))
                    for match in KEY_NUMBER_RE.finditer(sent.text)]
            keys = [k for k in keys if k]
            if not keys:
                continue
            for num in sent.numerals:
                if num.in_code:
                    continue
                stated = bool(PERCENT_TAIL_RE.match(
                    sent.slice(num.end, num.end + 20)))
                if num.fraction_word and not stated:
                    stated = True
                if not stated:
                    continue
                value = 100.0 * num.value if num.value <= 1.0 else num.value
                if not 0 < value <= 100:
                    continue
                # A percentage printed against its own parameter wins over the
                # sentence's named keys; otherwise the nearest named key does.
                tail = PERCENT_TAIL_RE.sub(
                    "", sent.slice(num.end, num.end + 45), count=1).split("%")[0]
                on_itself = AT_KEY_RE.search(tail)
                if on_itself:
                    key = on_itself.group(1)
                else:
                    key = min(keys, key=lambda k: abs(sent.text.find(k)
                                                      - (num.end - sent.start)))
                groups.setdefault(key, []).append(
                    (value, doc.path, line_of(doc, num.start), sent.text))
    findings = []
    for key, rows in sorted(groups.items()):
        values = sorted({round(r[0], 4) for r in rows})
        if len(values) < 2 or values[-1] / values[0] - 1.0 <= TOL_PLAIN:
            continue
        rows.sort()
        low, high = rows[0], rows[-1]
        finding = Finding("agreement", "mismatch", low[1], low[2],
                          f"{values[0]:g}% vs {values[-1]:g}%", low[3])
        finding.arithmetic = (
            f"key {key}: {values[0]:g} per cent at {low[1]}:{low[2]}, "
            f"{values[-1]:g} per cent at {high[1]}:{high[2]}")
        finding.detail = (
            "one quantity, two printed values for the same key. Neither site can "
            "see the other: only a check that reads both documents can.")
        findings.append(finding)
    return findings


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def published_set():
    docs = [REPO / name for name in PUBLISHED]
    docs_dir = REPO / "docs"
    if docs_dir.is_dir():
        docs.extend(sorted(docs_dir.glob("*.md")))
    return [p for p in docs if p.is_file()]


def order_findings(findings):
    return sorted(findings, key=lambda f: (SEVERITY.get(f.verdict, 9), f.path,
                                           f.line))


def report(findings, stream):
    current = None
    for finding in order_findings(findings):
        if finding.path != current:
            current = finding.path
            stream.write(f"\n== {current} ==\n")
        stream.write(f"[{finding.verdict.upper()}] {finding.check}  line "
                     f"{finding.line}\n")
        stream.write(f"  claim     : {finding.claim}\n")
        stream.write(f"  sentence  : {' '.join(finding.sentence.split())}\n")
        if finding.operands:
            stream.write("  operands  : " + ", ".join(
                f"{t} = {v:g} [{k}, from the {s}]" for (t, v, k, s) in finding.operands)
                + "\n")
        if finding.arithmetic:
            stream.write(f"  arithmetic: {finding.arithmetic}\n")
        if finding.detail:
            stream.write("  verdict   : " + " ".join(finding.detail.split()) + "\n")
    stream.write("\n")


def summarise(findings, stream):
    counts = {}
    for finding in findings:
        counts[finding.verdict] = counts.get(finding.verdict, 0) + 1
    stream.write("summary: " + ", ".join(f"{n} {verdict}"
                                         for verdict, n in sorted(counts.items()))
                 + "\n")


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

# (document, a literal that identifies the site, the verdict, the check).
# Every case is pinned to the calibration corpus rather than to a published
# document, because the published documents no longer carry a single one of
# these defects: each was repaired, and pinning a check to prose that has been
# made correct would leave it calibrated against nothing. The corpus reproduces
# the shape of each defect instead, so the check that caught it still reports
# the same verdict on an input that is still wrong.
CALIBRATION = (
    (CALIBRATION_DOC, "sixty times more accurate", "not derivable", "relation"),
    (CALIBRATION_DOC, "7.7 times as much", "mismatch", "relation"),
    (CALIBRATION_DOC, "five to eight orders of magnitude", "mismatch", "relation"),
    (CALIBRATION_DOC, "thousandths of a per cent", "mismatch", "relation"),
    (CALIBRATION_DOC, "0.9999 of the bound", "multiplier-unswept", "multiplier"),
    (CALIBRATION_DOC, "4.59", "mismatch", "agreement"),
    (CALIBRATION_DOC, "a factor of three", "mismatch", "relation"),
)


def selftest(stream):
    """Runs the checks over the calibration corpus and pins each verdict."""
    stream.write("calibration corpus\n")
    failures = 0
    cache = {}
    docs = {}
    for (relative, literal, verdict, check) in CALIBRATION:
        path = REPO / relative
        if not path.is_file():
            stream.write(f"  MISSING {relative}\n")
            failures += 1
            continue
        if relative not in docs:
            docs[relative] = build_doc(path)
        doc = docs[relative]
        if relative not in cache:
            cache[relative] = (check_relations(doc) + check_multiplier(doc)
                               + check_agreement([doc]))
        # A phrase the documents wrap mid-sentence reads with a newline where
        # the corpus literal has a space, so compare on normalised whitespace.
        found = [f for f in cache[relative]
                 if literal in " ".join(f.sentence.split())]
        match = [f for f in found if f.verdict == verdict and f.check == check]
        if match:
            stream.write(f"  ok   {relative}: \"{literal}\" -> {verdict} "
                         f"(line {match[0].line})\n")
        else:
            failures += 1
            seen = ", ".join(sorted({f.verdict + "/" + f.check for f in found})) \
                or "no finding mentions it"
            stream.write(f"  FAIL {relative}: \"{literal}\" -> wanted "
                         f"{verdict}/{check}, got {seen}\n")
    total = len(CALIBRATION)
    stream.write(f"\n{total - failures} of {total} calibration cases reproduced\n")
    return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Checks the arithmetic stated in the library's published prose.")
    parser.add_argument("files", nargs="*",
                        help="documents to scan; default the published set")
    parser.add_argument("--json", action="store_true", help="findings as JSON")
    parser.add_argument("--selftest", action="store_true",
                        help="run the calibration corpus and pin every verdict")
    parser.add_argument("--show-verified", action="store_true",
                        help="print the claims that did recompute")
    parser.add_argument("--strict", action="store_true",
                        help="also fail on multiplier-unswept findings")
    parser.add_argument("--min-confidence", choices=("high", "medium", "low"),
                        default="low",
                        help="the widest operand scope whose mismatch fails the run")
    args = parser.parse_args(argv)

    # The documents carry typographic dashes and minus signs; a console that
    # cannot encode them must not turn a finding into a traceback.
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

    if args.selftest:
        return selftest(sys.stdout)

    paths = [pathlib.Path(p) for p in args.files] or published_set()
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise SystemExit("check_doc_arithmetic: no such file: "
                         + ", ".join(str(p) for p in missing))
    docs = [build_doc(p) for p in paths]

    findings = []
    for doc in docs:
        findings.extend(check_relations(doc))
        findings.extend(check_multiplier(doc))
    findings.extend(check_agreement(docs))

    shown = [f for f in findings if args.show_verified or f.verdict != "verified"]
    if args.json:
        print(json.dumps([f.as_dict() for f in order_findings(shown)], indent=2))
    else:
        report(shown, sys.stdout)
        summarise(findings, sys.stdout)

    depth = {"high": 0, "medium": 1, "low": 2}[args.min_confidence]

    def failing(finding):
        if finding.verdict == "multiplier-unswept":
            return args.strict
        if finding.verdict != "mismatch":
            return False
        return finding.worst_scope() <= depth

    failures = [f for f in findings if failing(f)]
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
