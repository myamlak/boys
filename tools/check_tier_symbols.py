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
    one the linker can choose between is the failure section 2.2 of the design measured;
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

Exit status: 0 when no first-party function name is defined by both tiers, or when --check was
not asked for; 1 when one is, and 1 when an object cannot be read or holds no symbols.
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
          "tiers (intended: the tables are tier-independent and identical)")
    print(f"check_tier_symbols: {len(shared_other)} standard-library names are defined by both "
          "tiers (outside this project's token, reported and not asserted)")

    if not shared_functions:
        print(f"PASS: the tiers share no first-party function name "
              f"({len(baseline.functions)} against {len(vector.functions)} defined functions)")
        return 0

    print(f"check_tier_symbols: {len(shared_functions)} first-party function names are defined "
          "by both tiers")
    for name in shared_functions:
        print(f"  shared function: {name}")
    print(f"FAIL: {len(shared_functions)} first-party function names are defined by both tier "
          "object sets; the linker keeps one definition for the image, so which tier's "
          "instructions run is the link order's answer rather than the dispatcher's")
    return 1 if arguments.check else 0


if __name__ == "__main__":
    sys.exit(main())
