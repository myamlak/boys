#!/usr/bin/env python3
"""Generates .github/option-matrix.json, the configuration list CI runs.

The library carries several certified option axes (the multiply-add route, the
fit family, the evaluation scheme, the interval granularity, the arithmetic
precision, the host), and the accuracy gate judges each configuration they name
rather than one dimension of them. That list is data this script produces from
the library itself, so the workflow's `matrix:` is derived rather than
enumerated: adding a member to an axis is a regeneration, not a workflow edit.

Derivation, both directions. The members of each axis are read from the
declarations that carry them - the enumerations in include/boys/ and the
backend names the reports print - and every member read is required to appear
in the generated file. A member the library exposes and this table does not
account for is a hard error ("teach this script about it"), so an axis a lane
extends cannot land with CI silently still measuring the old set, and a
configuration cannot name a member the library does not serve.

WHY THE MATRIX IS OVER TWO AXES AND NOT SIX. The option space has six axes, and
a configuration is a point in it; the matrix is built over the axes a BUILD
realises, and the rest are recorded on every configuration rather than swept.
`device` and `route` are the realised pair: the host is the runner and toolchain
a tree is built in, and the multiply-add route is a property of the build rather
than of the source, so the tree that exercises the other route has to be built.
The other four are carried, and that is a fact about this revision rather than an
omission: `BoysFitRoutes()`, `BoysEvalSchemes()` and the backend table report
every member of `precision`, `family`, `evaluation` and `granularity` in a single
build, and the accuracy gate judges all of them in the one run it makes. Nothing
in the library - no CMake option, no compile definition, no gate flag - selects
one member of those four, so an entry per member would build the same tree twice
and assert nothing the other entry had not. Each configuration therefore records
the four as the members in force, and sweeping them becomes worthwhile exactly
when one of them acquires a build-time selection; a lane that adds one gives it a
realisation in this table and the axis becomes realised with it. The members of
all six axes are derived from the library either way, so a member added to a
carried axis reaches this file (and CI) by regeneration, with no workflow edit.

Usage:
    python tools/gen_option_matrix.py            # rewrite the option matrix
    python tools/gen_option_matrix.py --check    # fail if it is out of date
    python tools/gen_option_matrix.py --emit     # print the GitHub matrix
"""

import argparse
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
BACKEND_HPP = REPO / "include" / "boys" / "backend.hpp"
BOYS_HPP = REPO / "include" / "boys" / "boys.hpp"
SIMD_BACKEND_HPP = REPO / "src" / "boys_backend_simd.hpp"
CMAKELISTS = REPO / "CMakeLists.txt"
MATRIX_JSON = REPO / ".github" / "option-matrix.json"

FORMAT = "boys.option-matrix/1"

# The multiply-add route is the one axis a build selects, and this is the
# option that selects it. It is named here rather than read out of the
# workflow because the workflow is what consumes this file; that the option
# exists and defines the macro the headers read is checked against
# CMakeLists.txt below, so the name cannot go stale unnoticed.
ROUTE_OPTION = "BOYS_MULADD_SEPARATE"

# The hosts the option matrix configures on: the x86_64 legs this repository
# already runs, where the AVX2 tier is present and the gate's lane book is
# therefore judged at the cell count it is certified at. Each is checked
# against ci.yml's own legs - a host no leg builds on is a hard error.
# The compiler is named on the configure line rather than through CC/CXX: the
# matrix carries hosts with and without a compiler to name, and an empty CC in
# the environment is a different thing from no CC at all.
#
# `contract_guard` is what keeps a separate-route configuration from being
# vacuous, and it is the reason the two routes are two configurations rather
# than two names for one. The route in force is MEASURED, not declared:
# backend.hpp's RouteInForce() evaluates a bare `a * b + c` against the fused
# step and reports the fused route when the two agree, so on a build that
# contracts, BOYS_MULADD_SEPARATE=ON delivers the fused arithmetic and the
# entry would judge nothing. The guard states the property instead of
# inheriting it from the absence of an architecture flag, which is what it
# currently rests on and what a later global -march or /arch would take away.
# Every configuration that selects the separate route carries its host's guard.
HOSTS = (
    {"member": "linux-gcc", "runner": "ubuntu-latest", "toolchain": "gcc",
     "compiler": "-DCMAKE_CXX_COMPILER=g++", "generator": "",
     "contract_guard": "-ffp-contract=off"},
    {"member": "linux-clang", "runner": "ubuntu-latest", "toolchain": "clang",
     "compiler": "-DCMAKE_CXX_COMPILER=clang++", "generator": "",
     "contract_guard": "-ffp-contract=off"},
    # MSVC has no guard here because it needs none, and that was measured
    # rather than assumed. Contraction on this toolchain is OPT-IN: /fp:precise
    # (the default this tree builds with) does not contract, /fp:strict does not
    # either, and only /fp:fast or an explicit /fp:contract does. Measured on
    # MSVC 14.51.36231 at /O2 /arch:AVX2 by compiling the same bare
    # product-plus-add backend.hpp measures: /fp:precise -> does not contract,
    # /fp:precise /fp:contract -> contracts, /fp:strict -> does not contract,
    # /fp:fast -> contracts. /fp:strict would therefore be a heavier floating
    # point model than the property needs, and the tree does not build with one.
    # That is a measurement of one toolset and not a grant: the gate prints the
    # route in force for scalar-fp64 on every run, and the option-matrix job
    # runs it verbosely, so a runner whose compiler contracts anyway says so in
    # its log rather than passing quietly.
    {"member": "windows-msvc", "runner": "windows-latest", "toolchain": "msvc",
     "compiler": "", "generator": "-G Ninja", "contract_guard": ""},
)

# The axes, and where the members of each are declared. `build` axes are
# selected by a configuration; `carried` axes are in force in every build and
# judged by the accuracy gate in the same run.
AXES = (
    {"key": "device", "kind": "build",
     "source": "the x86_64 legs of .github/workflows/ci.yml"},
    {"key": "route", "kind": "build",
     "source": "backend.hpp: MulAddRoute"},
    {"key": "precision", "kind": "carried",
     "source": "the arithmetic backends' names"},
    {"key": "family", "kind": "carried",
     "source": "backend.hpp: FitRoute"},
    {"key": "evaluation", "kind": "carried",
     "source": "backend.hpp: EvalScheme"},
    {"key": "granularity", "kind": "carried",
     "source": "boys.hpp: EvalLane"},
)


def read(path):
    if not path.exists():
        raise SystemExit(f"gen_option_matrix: {path} is missing")
    return path.read_text(encoding="utf-8")


def enum_members(path, name):
    """The enumerators of `enum class <name>` in one header.

    A member is an identifier followed by `=`, `,` or the closing brace, which
    is what an enumerator of this tree's spelling always is. A header that
    stops carrying the declaration is a hard error rather than an empty set: a
    silent zero here would drop the whole axis.
    """
    text = read(path)
    match = re.search(
        r"enum\s+class\s+" + re.escape(name) + r"\b[^{]*\{(.*?)\n\};",
        text, re.DOTALL)
    if match is None:
        raise SystemExit(f"gen_option_matrix: no `enum class {name}` in {path} "
                         "- teach this script about it")
    body = re.sub(r"//[^\n]*", "", match.group(1))
    members = re.findall(r"\b(k[A-Za-z0-9_]+)\s*(?:=|,|\})", body)
    seen = []
    for member in members:
        if member not in seen:
            seen.append(member)
    if not seen:
        raise SystemExit(f"gen_option_matrix: `enum class {name}` in {path} "
                         "has no enumerators - teach this script about it")
    return seen


def kebab(name):
    """`kRationalMinimax` -> `rational-minimax`, `kRegionA` -> `region-a`."""
    text = name[1:] if name.startswith("k") else name
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "-", text)
    return text.lower()


def backend_names():
    """The arithmetic backend names the reports print, from their sources.

    The scalar pair's name is chosen in the template by the value type and the
    packed pair's is a literal; both spellings are read, so the two halves of
    the report's own table are the source rather than a copy of it.
    """
    names = set(re.findall(r'"(scalar-fp\d+|avx2-fp\d+)"', read(BACKEND_HPP)))
    names |= set(re.findall(r'"(scalar-fp\d+|avx2-fp\d+)"', read(SIMD_BACKEND_HPP)))
    if len(names) < 4:
        raise SystemExit("gen_option_matrix: the backend names are not all in "
                         "backend.hpp and src/boys_backend_simd.hpp - teach this "
                         "script about it")
    return sorted(names)


def route_option():
    """The option that selects the multiply-add route, checked in CMakeLists.

    The option has to exist, and has to define a macro the headers read, or the
    matrix would build a tree that reports the route it was not asked for.
    """
    text = read(CMAKELISTS)
    if not re.search(r"option\(\s*" + re.escape(ROUTE_OPTION) + r"\b", text):
        raise SystemExit(f"gen_option_matrix: CMakeLists.txt declares no "
                         f"{ROUTE_OPTION} option - teach this script about it")
    if not re.search(r"target_compile_definitions\(\s*boys\s+(?:PUBLIC|PRIVATE)\s+"
                     + re.escape(ROUTE_OPTION) + r"=1", text):
        raise SystemExit(f"gen_option_matrix: {ROUTE_OPTION} defines no macro on "
                         "the boys target - teach this script about it")
    return ROUTE_OPTION


def ci_legs():
    """Every CI leg of the workflow as (check name, runner, AVX2 tier).

    A leg records the tier it is held to as a matrix value where it has one and
    as the literal its configure step passes where it does not, and both
    spellings mean the same thing here. A leg asserted present is what a host
    is checked against; the runner's name alone decides nothing.
    """
    try:
        import yaml
    except ImportError:
        raise SystemExit("gen_option_matrix: reading the workflow's legs needs "
                         "pyyaml; the CI legs install it (pyyaml==6.0.2). "
                         "--emit does not, and is what the workflow runs.")

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import gen_platform_table as platform

    workflow = yaml.safe_load(platform.CI_YML.read_text(encoding="utf-8"))
    legs = []
    for job_id, job in workflow["jobs"].items():
        name = job.get("name", job_id)
        matrix = (job.get("strategy") or {}).get("matrix") or {}
        if not isinstance(matrix, dict):
            # A generated matrix: its rows are configurations, not hosts, and
            # reading them here would make this matrix its own input.
            continue
        literal = re.findall(r"-DBOYS_EXPECT_AVX2=(\d+)",
                             json.dumps(job.get("steps") or []))
        for entry in matrix.get("include") or [{}]:
            tier = entry.get("expect_avx2")
            if tier is None and literal:
                tier = int(literal[0])
            legs.append((platform.expand(name, entry),
                         platform.expand(job["runs-on"], entry), tier))
    return legs


def host_members():
    """The device axis's members, each checked against the workflow.

    A host is legal here only if the workflow already builds a leg on its
    runner whose dispatch assertion expects the AVX2 tier present: the lane
    book's cell count is a property of that tier, and an option-matrix job
    belongs on a host this repository has already shown runs it.
    """
    legs = ci_legs()
    members = []
    for host in HOSTS:
        found = [name for name, runner, avx2 in legs
                 if runner == host["runner"] and avx2 and int(avx2) != 0]
        if not found:
            raise SystemExit(
                f"gen_option_matrix: no leg of ci.yml builds on "
                f"{host['runner']} with the AVX2 tier present, so "
                f"{host['member']} is not a host this matrix may configure on")
        members.append(host["member"])
    return members


def axis_members():
    """Every axis's members, derived, with the build-selected ones marked."""
    names = backend_names()
    isas = sorted({name.split("-")[0] for name in names})
    widths = sorted({name.split("-")[1] for name in names})

    return {
        "device": host_members(),
        "route": [kebab(m) for m in enum_members(BACKEND_HPP, "MulAddRoute")],
        "precision": widths,
        "family": [kebab(m) for m in enum_members(BACKEND_HPP, "FitRoute")],
        "evaluation": [kebab(m) for m in enum_members(BACKEND_HPP, "EvalScheme")],
        "granularity": [kebab(m) for m in enum_members(BOYS_HPP, "EvalLane")],
        # The ISAs the backends name are the device's members on the library's
        # own terms; CI realises them as the host's runner, so they are carried
        # facts here rather than a second device axis.
        "_isa": isas,
    }


def configurations(members):
    """The allowed configurations: each host at each multiply-add route.

    The two build axes are the ones a configuration selects; the carried axes
    are in force in every build and the gate judges them in the same run, so
    they are recorded as the members they are rather than varied into jobs that
    would build the same tree twice.
    """
    route_option_name = route_option()
    out = []
    for host in HOSTS:
        for route in members["route"]:
            cmake = ["-DBOYS_EXPECT_AVX2=1",
                     f"-D{route_option_name}=" + ("ON" if route == "separate" else "OFF")]
            if host["compiler"]:
                cmake.append(host["compiler"])
            if route == "separate" and host["contract_guard"]:
                cmake.append("-DCMAKE_CXX_FLAGS=" + host["contract_guard"])
            out.append({
                "id": f"{host['member']}-{route}",
                "name": f"{host['member']} {route}",
                "runner": host["runner"],
                "toolchain": host["toolchain"],
                "generator": host["generator"],
                "build_jobs": "" if host["toolchain"] == "msvc" else "-j 4",
                "expect_avx2": 1,
                "cmake_args": " ".join(cmake),
                # The members this configuration names, per axis. A build axis
                # names one member; a carried axis names every member in force,
                # because the gate measures all of them in this one build.
                "selects": {
                    "device": host["member"],
                    "route": route,
                    "precision": members["precision"],
                    "family": members["family"],
                    "evaluation": members["evaluation"],
                    "granularity": members["granularity"],
                },
            })
    return out


def render():
    members = axis_members()
    members.pop("_isa")
    return {
        "format": FORMAT,
        "axes": [
            {"key": axis["key"], "kind": axis["kind"],
             "source": axis["source"], "members": members[axis["key"]]}
            for axis in AXES
        ],
        "configurations": configurations(members),
    }


def check_accounted(data, members):
    """Every member the library exposes must appear in the generated file.

    This is the direction that makes the file derived rather than transcribed:
    an axis a lane extends fails here until the table accounts for the new
    member, so a member cannot land while CI still measures the old set.
    """
    recorded = {axis["key"]: axis["members"] for axis in data["axes"]}
    for key, exposed in members.items():
        if key not in recorded:
            raise SystemExit(f"gen_option_matrix: the library exposes an axis "
                             f"{key!r} the option matrix does not carry - teach "
                             "this script about it")
        missing = [m for m in exposed if m not in recorded[key]]
        if missing:
            raise SystemExit(
                f"gen_option_matrix: the library exposes {key} member(s) "
                f"{missing} that the option matrix does not account for - "
                "regenerate it, and give each new member a realisation and a "
                "configuration if it has one")


def emit(data):
    """The GitHub matrix object: the job inputs, and nothing else."""
    keys = ("name", "runner", "toolchain", "generator", "build_jobs",
            "cmake_args")
    return {"include": [{key: c[key] for key in keys}
                        for c in data["configurations"]]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="exit nonzero if the option matrix is out of date")
    parser.add_argument("--emit", action="store_true",
                        help="print the GitHub matrix object and exit")
    args = parser.parse_args()

    if args.emit:
        # From the committed file, not from a fresh derivation: the workflow
        # expands what is committed, and --check on the build legs is what
        # proves that file is what this revision generates. It also keeps the
        # scheduling job free of every dependency the derivation has.
        if not MATRIX_JSON.exists():
            raise SystemExit(f"gen_option_matrix: {MATRIX_JSON} is missing")
        committed = json.loads(MATRIX_JSON.read_text(encoding="utf-8"))
        if committed.get("format") != FORMAT:
            raise SystemExit(f"gen_option_matrix: {MATRIX_JSON} is not a "
                             f"{FORMAT} file")
        json.dump(emit(committed), sys.stdout, separators=(",", ":"))
        sys.stdout.write("\n")
        return 0

    members = axis_members()
    exposed = dict(members)
    exposed.pop("_isa")
    data = render()

    text = json.dumps(data, indent=2) + "\n"

    if args.check:
        if not MATRIX_JSON.exists():
            print(f"DRIFT: {MATRIX_JSON} is missing", file=sys.stderr)
            return 1
        current = MATRIX_JSON.read_text(encoding="utf-8")
        committed = json.loads(current)
        check_accounted(committed, exposed)
        if current != text:
            print("DRIFT: .github/option-matrix.json is not what the library "
                  "generates; run tools/gen_option_matrix.py", file=sys.stderr)
            return 1
        print(f"option matrix matches the library "
              f"({len(data['configurations'])} configurations)")
        return 0

    check_accounted(data, exposed)
    MATRIX_JSON.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {len(data['configurations'])} configurations into "
          f"{MATRIX_JSON.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
