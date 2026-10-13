#!/usr/bin/env python3
"""A translation unit that reads a seam compiles that seam's rows and says so.

WHAT IT CHECKS. One probe, compiled and LINKED twice with NO archive on the link line - the front
door alone, which is the contract now that the default entries are inline in the headers - and run:

  * against this build's own seam: the headers as they stand. The committed table is the default
    reading; a replaced build's own seam is read by handing the tool that build's own include
    directory and the BOYS_BUILD_DEFAULTS_* defines through --include and --define;
  * against a replaced seam: a second defaults header, copied into a directory that precedes the
    include directories on the search path, with the defines a configure puts on a build that
    replaces its table - BOYS_BUILD_DEFAULTS_REPLACED=1 and an identity named from the file's own
    sha256, which is the name CMakeLists.txt derives the same way.

Each run prints, from inside its own translation unit, the seam it compiled into - the namespace
the front door declares its default entries in, stringized - which of the two tables it read, and a
small digest of the default entries' bits over a few fixed points, F_3(1.25) among them. Four
readings come back:

  * both probes build and run, so a translation unit that reads a seam needs no archive to satisfy
    its call;
  * each prints the seam name it was built for, so the seam a unit read is observable at runtime;
  * the two digests differ, so the replaced table's rows did move the values - a digest that agrees
    is a finding, and the tool says which of the two causes it can be;
  * the replaced reading at F_3(1.25) is the value the replaced table prints, handed in beside it.

The two probes compile under the same flags, so a digest difference is the seams' and not the
flags'. The entries are the caller's to compile, and the profile the
certified figures belong to is `docs/build-facts.md`'s business rather than this check's.

    python tools/check_seam_identity.py --check --cxx g++ --std=-std=c++20 \\
        --include include --replacement tests/build_defaults_tuned.hpp \\
        --replaced-value 0.055476132923077695
    python tools/check_seam_identity.py --cxx g++ --include include \\
        --replacement ... --replaced-value ...                 # report only

The caller hands one argument of the build's own flags through `--flag=ARG` per argument (the `=`
form, so a flag beginning with `-` is a value and not an option): both probes then compile the way
that configuration compiles, and the digests stay the configuration's rather than the tool's.

Exit status: 0 when both probes build, print the seam they were built for, print two different
digests, and the replaced reading at F_3(1.25) is the expected value, and 0 when --check was not
asked for; 1 otherwise, and 1 whenever a probe cannot be built or run.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

SOURCE = """#include <boys/boys.hpp>

#include <bit>
#include <cstdint>
#include <cstdio>

// The seam's own name: the identifier the front door names its default entries' namespace by.
#define BOYS_CHECK_STRINGIZE_1(text) #text
#define BOYS_CHECK_STRINGIZE(text) BOYS_CHECK_STRINGIZE_1(text)

namespace {

/// FNV-1a over the bits of every value the entries below returned.
std::uint64_t gDigest = 14695981039346656037ull;

void Mix(double value) {
    gDigest ^= std::bit_cast<std::uint64_t>(value);
    gDigest *= 1099511628211ull;
}

} // namespace

int main() {
    std::printf("seam %s\\n", BOYS_CHECK_STRINGIZE(BOYS_DETAIL_SEAM_NAMESPACE));
    // The committed table's own header defines BOYS_BUILD_DEFAULTS_COMMITTED and a replacement
    // does not, so a unit says which of the two tables its entries came from.
#if defined(BOYS_BUILD_DEFAULTS_COMMITTED)
    std::printf("table committed\\n");
#else
    std::printf("table replaced\\n");
#endif

    // A few fixed points across the argument regions, one of them F_3(1.25): a table whose rows
    // moved prints other bits here, and the reading at F_3(1.25) is printed on its own so a
    // person can compare it against the number the table is documented to print.
    const double points[] = {0.25, 1.25, 2.5, 10.0, 30.0};
    for (double x : points)
    {
        for (int n = 0; n <= 8; ++n)
        {
            Mix(boys::BoysSingle(n, x));
        }
        double ladder[9] = {};
        boys::BoysAllOrders(8, x, ladder);
        for (double value : ladder)
        {
            Mix(value);
        }
    }

    std::printf("F3(1.25) %.17g\\n", boys::BoysSingle(3, 1.25));
    std::printf("digest %016llx\\n", static_cast<unsigned long long>(gDigest));
    return 0;
}
"""

SEAM_PREFIX = "seam_"
REPLACED_DEFINES = ("BOYS_BUILD_DEFAULTS_REPLACED=1",)


def compile_and_run(cxx: str, std: str, includes: list[Path], defines: list[str],
                    flags: list[str], source: Path, work: Path, name: str) -> list[str] | int:
    """One probe's output lines, or the process's exit status when it did not build or run."""
    binary = work / name
    command = [cxx, std, *[f"-I{directory}" for directory in includes],
               *[f"-D{define}" for define in defines], *flags, str(source), "-o", str(binary)]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        print(f"{name}: did not build - {cxx} exited {completed.returncode}")
        print(((completed.stderr or completed.stdout).strip().splitlines() or [""])[0][:300])
        return completed.returncode
    run = subprocess.run([str(binary)], capture_output=True, text=True)
    if run.returncode != 0:
        print(f"{name}: built, but exited {run.returncode}; printed {run.stdout.strip()!r}")
        return run.returncode
    return run.stdout.splitlines()


def reading(name: str, lines: list[str]) -> dict[str, str]:
    """One probe's printed reading, by field."""
    fields: dict[str, str] = {}
    for line in lines:
        key, _, value = line.partition(" ")
        fields[key] = value
    print(f"  {name}: seam {fields.get('seam', 'nothing')}, "
          f"{fields.get('table', 'nothing')} table, "
          f"F3(1.25) {fields.get('F3(1.25)', 'nothing')}, "
          f"digest {fields.get('digest', 'nothing')}")
    return fields


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero unless the two probes read the seams they were built "
                             "for, printed two different digests, and the replaced table prints "
                             "the expected value")
    parser.add_argument("--cxx", required=True)
    parser.add_argument("--std", default="-std=c++20")
    parser.add_argument("--include", required=True, action="append", type=Path, metavar="DIR",
                        help="one include directory of the build's own seam, in search order; "
                             "repeats, and the first is searched first")
    parser.add_argument("--define", action="append", default=[], metavar="NAME=VALUE",
                        help="one define the build's own seam is read under (a replaced build "
                             "carries BOYS_BUILD_DEFAULTS_REPLACED=1 and "
                             "BOYS_BUILD_DEFAULTS_SEAM_IDENT=seam_<sha256>); the replaced probe "
                             "drops these and names the replacement's own seam")
    parser.add_argument("--seam-ident", default=f"{SEAM_PREFIX}committed",
                        help="the identity the own-seam probe must print")
    parser.add_argument("--replacement", required=True, type=Path,
                        help="the second defaults header: a replacement for "
                             "boys/boys_build_defaults.hpp whose rows move the values")
    parser.add_argument("--replaced-value", required=True,
                        help="the value the replaced table prints at F_3(1.25)")
    parser.add_argument("--flag", action="append", default=[], metavar="ARG",
                        help="one argument of the build's own flags, repeated; applied to both "
                             "probes' compile and link, so the digests are the configuration's")
    arguments = parser.parse_args()

    if not arguments.replacement.is_file():
        print(f"check_seam_identity: no replacement header at {arguments.replacement}",
              file=sys.stderr)
        return 1
    # The identity a configure gives a replaced table: its own sha256, as a namespace token
    # (CMakeLists.txt derives the same name from the same bytes).
    replaced_ident = SEAM_PREFIX + hashlib.sha256(arguments.replacement.read_bytes()).hexdigest()

    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        source = work / "seam_probe.cpp"
        source.write_text(SOURCE, encoding="utf-8")
        # The replacement under the name the headers include, in a directory searched before every
        # own include directory: the compiler reads it as boys/boys_build_defaults.hpp and never
        # reaches the committed file, which is how a build that replaces its table is configured.
        (work / "boys").mkdir()
        (work / "boys" / "boys_build_defaults.hpp").write_bytes(
            arguments.replacement.read_bytes())

        own = compile_and_run(arguments.cxx, arguments.std, arguments.include, arguments.define,
                              arguments.flag, source, work, "own-seam-probe")
        replaced_defines = [define for define in arguments.define
                            if not define.startswith("BOYS_BUILD_DEFAULTS")]
        replaced = compile_and_run(
            arguments.cxx, arguments.std, [work, *arguments.include],
            [*replaced_defines, *REPLACED_DEFINES,
             f"BOYS_BUILD_DEFAULTS_SEAM_IDENT={replaced_ident}"],
            arguments.flag, source, work, "replaced-seam-probe")

    if isinstance(own, int) or isinstance(replaced, int):
        return 1

    print(f"own seam ({arguments.seam_ident}) and replacement ({arguments.replacement}):")
    own_fields = reading("own seam", own)
    replaced_fields = reading("replaced seam", replaced)

    failures: list[str] = []
    if own_fields.get("seam") != arguments.seam_ident:
        failures.append(f"the own-seam probe read seam {own_fields.get('seam')!r}, not "
                        f"{arguments.seam_ident!r}")
    expected_own_table = "committed" if arguments.seam_ident == f"{SEAM_PREFIX}committed" else None
    if expected_own_table and own_fields.get("table") != expected_own_table:
        failures.append(f"the own-seam probe read the {own_fields.get('table')} table, and the "
                        f"seam {arguments.seam_ident!r} names the {expected_own_table} one")
    if replaced_fields.get("seam") != replaced_ident:
        failures.append(f"the replaced-seam probe read seam {replaced_fields.get('seam')!r}, not "
                        f"the replacement's own {replaced_ident!r}")
    if replaced_fields.get("table") != "replaced":
        failures.append(f"the replaced-seam probe read the {replaced_fields.get('table')} table, "
                        f"so {arguments.replacement} is not the file it compiled")
    if own_fields.get("digest") == replaced_fields.get("digest"):
        failures.append("the two probes printed one digest: either the replacement's rows did not "
                        "move the values, or both probes read the same table")
    if not same_value(replaced_fields.get("F3(1.25)"), arguments.replaced_value):
        failures.append(f"the replaced table prints {replaced_fields.get('F3(1.25)')!r} at "
                        f"F_3(1.25), not {arguments.replaced_value!r}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1 if arguments.check else 0

    print(f"PASS: a translation unit reading each seam compiled that seam's rows - "
          f"{arguments.seam_ident} and {replaced_ident} - and the replaced table's rows moved the "
          f"digest and the value at F_3(1.25)")
    return 0


def same_value(printed: str | None, expected: str) -> bool:
    """Whether two printed decimals are the same value; the comparison is exact, not rounded."""
    try:
        return printed is not None and float(printed) == float(expected)
    except ValueError:
        return False


if __name__ == "__main__":
    sys.exit(main())
