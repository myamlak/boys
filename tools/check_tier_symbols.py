#!/usr/bin/env python3
"""The two carried tiers are two sets of object files, and no function may live in both.

WHAT IT CHECKS. A dispatch build compiles the library's bodies twice, once per tier, and links
both into one image. A name defined by both tiers is one symbol in that image: the linker keeps
whichever definition it met first, so the tier the processor selected is not the tier whose body
runs, and on a machine without AVX2 the merged choice is an illegal instruction. The per-tier
inline namespace token exists to give every tier-varying definition its own mangled name; this
tool is the regression test for that property - it reads the symbol table of each tier's object
files and fails when a first-party function name is defined by both.

What is compared, and why the scope is stated rather than assumed:

  * first-party function names are the assertion. A function carries executable instructions, and
    one the linker can choose between is the failure section 2.2 of the design measured. The one
    sharing accepted is the literal allowlist below: members of the public half-format wrapper
    classes, whose mangled name carries their class and whose class is the API. The observed
    shared set must equal that allowlist, filtered by the wrapper types this build compiles, so
    a new shared name fails and so does an entry the tiers do not share;
  * first-party data names are reported but not asserted. The coefficient tables are
    tier-independent: both copies come from the same header and hold the same bytes, so the
    linker's choice between them changes nothing that executes, and token-namespacing them would
    put a second copy of every table in the image;
  * standard-library and C-runtime names are reported but not asserted. Their inline definitions
    are emitted by the compiler under this project's tier flags and cannot be given a boys token;
    a correct implementation of the token cannot make them disjoint either.

The names are read in the platform's own form and never demangled: a token is part of the mangled
name, and a demangler that drops or reorders namespaces would hide exactly the difference this
check reads. A tier object with no symbol table is a failure and not a pass: a check over nothing
has found nothing.

Two symbol-table readers are supported, the ones this tree's own build-facts probe uses:

    python tools/check_tier_symbols.py --check --mode msvc-dump --tool link.exe \\
        --baseline base1.obj base2.obj --avx2 avx1.obj avx2.obj
    python tools/check_tier_symbols.py --check --mode nm --tool nm \\
        --baseline base1.o base2.o --avx2 avx1.o avx2.o
    python tools/check_tier_symbols.py ...                    # report only

`msvc-dump` runs `<tool> -dump -symbols` (link.exe carries dumpbin's dump mode on every MSVC
install) and reads the COFF symbol table: a symbol with a section index and the External storage
class is defined, a function is the `()`-marked record, and an `UNDEF` record is a reference and
not a definition. `nm` runs `<tool> --defined-only -g`: an uppercase type is a definition, `T`/`W`
are functions and `D`/`B`/`R`/`S`/`G`/`V`/`C` are data.

Exit status: 0 when the shared first-party function names equal the allowlist below, or when
--check was not asked for; 1 when they do not, and 1 when an object cannot be read or holds no
symbols.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# One COFF symbol record: index, value, where it lives, what it is, its storage class, its name.
# A record with a section index and the External storage class is a definition; `UNDEF` is a
# reference, and a `Static` or `Label` record is not an external name at all.
MSVC_SYMBOL = re.compile(
    r"^\s*[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(UNDEF|SECT[0-9A-Fa-f]+|ABS)\s+"
    r"notype(\s+\(\))?\s+(External|WeakExternal|Static|Label)\s+\|\s+(\S+)")

# One nm line: address, type, name. A line without an address is still a symbol.
NM_SYMBOL = re.compile(r"^\s*(?:[0-9A-Fa-f]+)?\s*([A-Za-z?])\s+(\S+)\s*$")

NM_FUNCTIONS = "TW"
NM_DATA = "DBRSGVC"

# The first-party functions this check accepts as defined by both tiers, and why each cannot be
# given the tier token. Every one is a member of the public F16 or Bf16 wrapper class: its
# mangled name carries the class, and the class is the public type, so tokenizing the member
# would make boys::F16 two types, one per tier. The definitions are constexpr members of a class
# defined in a header, and a constexpr member must be defined wherever it is constant-evaluated,
# so they cannot move to one tier-independent unit either. Tier flags cannot change what they
# compute: every body is integer bit manipulation (bit_cast, masks, shifts, integer compares)
# with exact, rounding-free semantics, and the forced-tier digest covers the half lanes
# (tests/entry_digest.hpp, SweepHalf).
#
# The set is asserted by equality, not by membership: the observed shared set must equal the
# entries whose wrapper type this build compiles. A type is compiled when a symbol carrying its
# class name is defined by either tier set; a toolchain with std::float16_t has no wrapper class
# and so expects no entry. An entry whose type is compiled while the name is not shared is a
# stale allowlist and fails the same way a new shared name does.
HALF_MEMBERS = {
    "??0F16@boys@@QEAA@M@Z": (
        "F16", "F16(float): integer masks and shifts through FloatToHalf; no FP operation"),
    "??0F16@boys@@QEAA@XZ": (
        "F16", "F16(): value-initialises the bit pattern; no arithmetic"),
    "??BF16@boys@@QEBAMXZ": (
        "F16", "operator float: HalfToFloat builds the float bits and bit_casts; exact"),
    "?FloatToHalf@F16@boys@@CAGM@Z": (
        "F16", "FloatToHalf: bit_cast, masks, shifts, integer compares; no FP rounding"),
    "?HalfToFloat@F16@boys@@CAMG@Z": (
        "F16", "HalfToFloat: integer shifts into a float bit pattern, bit_cast; exact"),
    "??0Bf16@boys@@QEAA@M@Z": (
        "Bf16", "Bf16(float): integer rounding and masks through FloatToBf16; no FP operation"),
    "??0Bf16@boys@@QEAA@XZ": (
        "Bf16", "Bf16(): value-initialises the bit pattern; no arithmetic"),
    "??BBf16@boys@@QEBAMXZ": (
        "Bf16", "operator float: Bf16ToFloat shifts the bits and bit_casts; exact"),
    "?FloatToBf16@Bf16@boys@@CAGM@Z": (
        "Bf16", "FloatToBf16: mantissa rounded with integer arithmetic and masks; no FP operation"),
    "?Bf16ToFloat@Bf16@boys@@CAMG@Z": (
        "Bf16", "Bf16ToFloat: one shift into a float bit pattern, bit_cast; exact"),
}

# How a compiled wrapper type is recognised: its class name inside a mangled boys symbol. The
# names above are the MSVC manglings; a non-MSVC toolchain that compiles the wrapper (it has no
# std::float16_t) spells them differently, so the check fails rather than passing silently, and
# its spellings belong here once such a build registers the test.
HALF_TYPE_MARKERS = {"F16": "F16@boys@@", "Bf16": "Bf16@boys@@"}


class Symbols:
    """One tier's defined external names, split into functions and data."""

    def __init__(self, functions: set[str], data: set[str]) -> None:
        self.functions = functions
        self.data = data

    def names(self) -> set[str]:
        return self.functions | self.data


def read_msvc(tool: str, path: Path) -> Symbols:
    completed = subprocess.run([tool, "-dump", "-symbols", str(path)],
                               capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"{path.name}: {tool} -dump -symbols exited "
                           f"{completed.returncode}: {completed.stderr.strip()[:300]}")
    functions: set[str] = set()
    data: set[str] = set()
    for line in completed.stdout.splitlines():
        match = MSVC_SYMBOL.match(line)
        if not match:
            continue
        section, is_function, storage, name = match.groups()
        if section == "UNDEF" or storage not in ("External", "WeakExternal"):
            continue
        (functions if is_function else data).add(name)
    return Symbols(functions, data)


def read_nm(tool: str, path: Path) -> Symbols:
    completed = subprocess.run([tool, "--defined-only", "-g", str(path)],
                               capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"{path.name}: {tool} --defined-only -g exited "
                           f"{completed.returncode}: {completed.stderr.strip()[:300]}")
    functions: set[str] = set()
    data: set[str] = set()
    for line in completed.stdout.splitlines():
        match = NM_SYMBOL.match(line)
        if not match:
            continue
        kind, name = match.groups()
        if kind in NM_FUNCTIONS:
            functions.add(name)
        elif kind in NM_DATA:
            data.add(name)
    return Symbols(functions, data)


READERS = {"msvc-dump": read_msvc, "nm": read_nm}


def object_paths(spelled: list[str]) -> list[Path]:
    """The object files one argument names.

    A test command hands a member's `$<TARGET_OBJECTS:...>` as ONE argument holding the list
    separated by semicolons, because the generator expression is expanded inside the argument and
    not into new arguments; a person copying the command out of the test file gets the same form.
    Both spellings name the same set, so both are read here.
    """
    paths = [Path(part) for argument in spelled for part in argument.split(";") if part]
    return paths


def read_tier(reader, tool: str, paths: list[Path]) -> Symbols:
    """The union of every object's defined names, with a tool failure leaving through."""
    functions: set[str] = set()
    data: set[str] = set()
    for path in paths:
        symbols = reader(tool, path)
        functions |= symbols.functions
        data |= symbols.data
    return Symbols(functions, data)


def is_first_party(name: str) -> bool:
    """Whether the mangled name names this library: the token is a boys namespace, and a
    standard-library or C-runtime definition carries none."""
    return "boys" in name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero when a first-party function name is defined by both "
                             "tiers")
    parser.add_argument("--mode", choices=sorted(READERS), required=True,
                        help="how to read an object's symbol table")
    parser.add_argument("--tool", required=True,
                        help="the symbol-table reader: link.exe for msvc-dump, nm for nm")
    parser.add_argument("--baseline", nargs="+", required=True,
                        help="the baseline tier's object files")
    parser.add_argument("--avx2", nargs="+", required=True,
                        help="the vector tier's object files")
    arguments = parser.parse_args()

    baseline_paths = object_paths(arguments.baseline)
    vector_paths = object_paths(arguments.avx2)
    reader = READERS[arguments.mode]
    try:
        baseline = read_tier(reader, arguments.tool, baseline_paths)
        vector = read_tier(reader, arguments.tool, vector_paths)
    except RuntimeError as failure:
        print(f"check_tier_symbols: {failure}")
        return 1

    if not baseline.names() or not vector.names():
        print("check_tier_symbols: a tier object set yielded no defined external names "
              f"(baseline {len(baseline.names())}, vector {len(vector.names())}); a check over "
              "nothing is not a pass")
        return 1

    print(f"check_tier_symbols: baseline {len(baseline_paths)} objects, "
          f"{len(baseline.functions)} functions and {len(baseline.data)} data names")
    print(f"check_tier_symbols: vector {len(vector_paths)} objects, "
          f"{len(vector.functions)} functions and {len(vector.data)} data names")

    shared_functions = sorted(name for name in baseline.functions & vector.functions
                              if is_first_party(name))
    shared_data = sorted(name for name in baseline.data & vector.data
                         if is_first_party(name))
    shared_other = sorted((baseline.names() & vector.names())
                          - set(shared_functions) - set(shared_data))

    print(f"check_tier_symbols: {len(shared_data)} first-party data names are defined by both "
          "tiers (reported, not asserted: the tier-independent tables hold identical bytes, and "
          "the one guard registry is an inline object the linker merges)")
    print(f"check_tier_symbols: {len(shared_other)} standard-library names are defined by both "
          "tiers (outside this project's token, reported and not asserted)")

    # The allowlist is the assertion, not a filter: the observed set must equal the entries whose
    # wrapper type this build compiles, so a new shared name and a stale entry are both loud.
    every_name = baseline.names() | vector.names()
    compiled_types = {type_name for type_name, marker in HALF_TYPE_MARKERS.items()
                      if any(marker in name for name in every_name)}
    expected = {name for name, (type_name, _) in HALF_MEMBERS.items()
                if type_name in compiled_types}

    observed = set(shared_functions)
    shared_allowed = sorted(observed & set(HALF_MEMBERS))
    unexpected = sorted(observed - set(HALF_MEMBERS))
    stale = sorted(expected - observed)

    if shared_allowed:
        print(f"check_tier_symbols: {len(shared_allowed)} first-party function names are shared "
              "by both tiers and allowlisted (the public half-format wrappers)")
        for name in shared_allowed:
            print(f"  allowlisted: {name} - {HALF_MEMBERS[name][1]}")

    if unexpected or stale:
        for name in unexpected:
            print(f"  unexpected shared function: {name}")
        for name in stale:
            print(f"  stale allowlist entry: {name} - its wrapper type is compiled in this "
                  "build but the two tiers no longer share this definition")
        print(f"FAIL: the shared first-party function set does not equal the allowlist "
              f"({len(observed)} shared, {len(expected)} expected); a shared name the linker can "
              "choose between is the failure the token exists to prevent, and an allowlist that "
              "has drifted is a check that no longer checks")
        return 1 if arguments.check else 0

    if not observed:
        print(f"PASS: the tiers share no first-party function name "
              f"({len(baseline.functions)} against {len(vector.functions)} defined functions)")
        return 0

    print(f"PASS: the shared first-party function set equals the allowlist "
          f"({len(shared_allowed)} shared, every one a public half-wrapper member)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
