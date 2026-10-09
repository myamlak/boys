#!/usr/bin/env python3
"""Two libraries built under different seams are not interchangeable, and the linker says so.

WHAT IT CHECKS. The front door defines the entries a call site names no policy for in a namespace
named from the seam the library was configured with, so a translation unit compiled against
another seam asks for symbols no library built with that seam carries. This tool compiles one
call against two seam identities and links each against the library:

  * the build's OWN seam must link - the control, without which a probe that failed for its own
    reasons (a wrong archive, a missing symbol for something else) would read as a refusal;
  * another seam must NOT link.

    python tools/check_seam_identity.py --check --cxx g++ --include include \\
        --library build/libboys.a --seam-ident seam_committed

Exit status: 0 when the control links and the other seam does not, and when --check was not asked
for; 1 otherwise, and 1 when the two conditions are not both read.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

SOURCE = """#include <boys/boys.hpp>

int main() {
    double out[33] = {};
    boys::BoysAllOrders(0, 1.0, out);
    return out[0] == 0.0 ? 0 : 1;
}
"""

OTHER_SEAM = "seam_other_than_this_build"


def link(cxx: str, std: str, include: Path, library: Path, work: Path, define: str | None):
    """Compile and link the call, with or without a seam identity of its own."""
    source = work / "seam_probe.cpp"
    source.write_text(SOURCE, encoding="utf-8")
    binary = work / "seam_probe"
    command = [cxx, std, f"-I{include}"]
    if define:
        command.append(f"-DBOYS_BUILD_DEFAULTS_SEAM_IDENT={define}")
    command += [str(source), str(library), "-o", str(binary)]
    completed = subprocess.run(command, capture_output=True, text=True)
    return completed.returncode, (completed.stderr or completed.stdout).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero unless the control links and the other seam does not")
    parser.add_argument("--cxx", required=True)
    parser.add_argument("--std", default="-std=c++20")
    parser.add_argument("--include", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--seam-ident", default="seam_committed",
                        help="the identity this build's own seam is named by")
    arguments = parser.parse_args()

    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        own_status, own_error = link(arguments.cxx, arguments.std, arguments.include,
                                     arguments.library, work, arguments.seam_ident)
        other_status, other_error = link(arguments.cxx, arguments.std, arguments.include,
                                         arguments.library, work, OTHER_SEAM)

    print(f"own seam ({arguments.seam_ident}): {'linked' if own_status == 0 else 'did not link'}")
    print(f"another seam ({OTHER_SEAM}): {'linked' if other_status == 0 else 'did not link'}")
    if other_status != 0:
        first = (other_error.splitlines() or [""])[0]
        print(f"  the linker said: {first[:200]}")
    if own_status != 0:
        print(f"  the control failed: {(own_error.splitlines() or [''])[0][:200]}")

    if own_status == 0 and other_status != 0:
        print("PASS: this library serves the seam it was built with and refuses another")
        return 0

    if own_status != 0:
        print("FAIL: a translation unit compiled against this build's own seam does not link it, "
              "so the refusal below is not the one this check is about")
    else:
        print("FAIL: a translation unit compiled against another seam linked this library, so "
              "two libraries built under different seams can be interchanged without the linker "
              "saying so")
    return 1 if arguments.check else 0


if __name__ == "__main__":
    sys.exit(main())
