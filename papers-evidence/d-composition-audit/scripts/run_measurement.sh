#!/usr/bin/env bash
# Run the composed-bounds interaction measurement on the second target: GCC on
# Linux, inside WSL. This is the cross-target half of the stability design; the
# Windows/MSVC half is run_measurement.ps1.
#
# Machine rule (coordinator): this is a timing-sensitive measurement. Do not run
# it until the coordinator signals. Run it on a quiet machine, alone: no build,
# lane, generator or other timing run concurrently. The driver takes a canary
# gate before the first cell and reads the canary again after the last; a wide
# gate stops the run. In WSL the canary sees the guest's own noise only, so a
# wide gate with an idle guest means the polluter is on the Windows side: sample
# the host's process deltas rather than assuming the guest is busy.
#
# The build, argument emission and self-tests happen before the canary gate.
# The reference is not regenerated here: the second target measures the same
# narrowed argument grid, so it reuses the primary run's 60-digit reference and
# refuses if the emitted arguments differ from it. This keeps mpmath out of the
# WSL dependency set, where it may not be installed.
# Under the owner's 2026-10-20 ruling the canary is a hint: with
# BOYS_CANARY_HINT_ONLY=1 a wide reading is recorded and the run proceeds under
# AMENDMENT-2026-10-20, with its rows labelled timing-provisional.
#
# Usage, only after the coordinator's signal:
#   bash scripts/run_measurement.sh --coordinator-signal \
#       --reference results/reference.csv \
#       --arguments-reference results/arguments.csv
#   bash scripts/run_measurement.sh --coordinator-signal --route separate \
#       --reference results/reference.csv \
#       --arguments-reference results/arguments.csv
#
# Both paths are relative to papers/composed-bounds when run from there. When
# the paper checkout is a Windows worktree whose gitdir WSL cannot resolve, pass
# --papers-revision and --papers-dirty with the values read on the host.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
BOUNDS_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

BOYS_SOURCE="/mnt/c/Users/myaml/dev/bladawiec/external/boys"
EXPECTED_REVISION="482fd3388c276091834a6c12a642d9de2481c9f2"
ROUTE="fused"
OUT_DIR=""
BUILD_DIR=""
REFERENCE=""
ARGUMENTS_REFERENCE=""
PYTHON="python3"
PAPERS_REVISION=""
PAPERS_DIRTY=""
ALLOW_OTHER_REVISION=0
COORDINATOR_SIGNAL=0

usage() {
    sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
    case "$1" in
        --route) ROUTE="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --reference) REFERENCE="$2"; shift 2 ;;
        --arguments-reference) ARGUMENTS_REFERENCE="$2"; shift 2 ;;
        --boys-source) BOYS_SOURCE="$2"; shift 2 ;;
        --expected-revision) EXPECTED_REVISION="$2"; shift 2 ;;
        --python) PYTHON="$2"; shift 2 ;;
        --papers-revision) PAPERS_REVISION="$2"; shift 2 ;;
        --papers-dirty) PAPERS_DIRTY="$2"; shift 2 ;;
        --allow-other-revision) ALLOW_OTHER_REVISION=1; shift ;;
        --coordinator-signal) COORDINATOR_SIGNAL=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ "$COORDINATOR_SIGNAL" -ne 1 ]; then
    echo "run_measurement: wait for the coordinator's signal: the interaction probe is" >&2
    echo "timing-sensitive and runs on a quiet machine, alone. Pass" >&2
    echo "--coordinator-signal only when it has been given." >&2
    exit 2
fi

case "$ROUTE" in
    fused|separate) ;;
    *) echo "run_measurement: --route must be fused or separate" >&2; exit 2 ;;
esac

if [ -z "$OUT_DIR" ]; then
    OUT_DIR="$BOUNDS_DIR/results-gcc-$ROUTE"
fi
if [ -z "$BUILD_DIR" ]; then
    BUILD_DIR="$BOUNDS_DIR/build-gcc-$ROUTE"
fi
if [ -z "$REFERENCE" ]; then
    echo "run_measurement: the second target reuses the primary run's reference at the" >&2
    echo "same narrowed inputs; pass --reference results/reference.csv" >&2
    exit 2
fi
if [ ! -f "$REFERENCE" ]; then
    echo "run_measurement: reference '$REFERENCE' does not exist" >&2
    exit 2
fi

if [ ! -f "$BOYS_SOURCE/CMakeLists.txt" ]; then
    echo "run_measurement: '$BOYS_SOURCE' does not look like the boys source" >&2
    exit 2
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "run_measurement: '$PYTHON' is not on PATH" >&2
    exit 2
fi
for tool in cmake ninja git cmp; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "run_measurement: '$tool' is not on PATH" >&2
        exit 2
    fi
done

gitsafe=(git -c "safe.directory=$BOYS_SOURCE" -C "$BOYS_SOURCE")
revision=$("${gitsafe[@]}" rev-parse HEAD)
dirty=$("${gitsafe[@]}" status --porcelain | wc -l)
if [ "$revision" != "$EXPECTED_REVISION" ] && [ "$ALLOW_OTHER_REVISION" -ne 1 ]; then
    echo "run_measurement: boys is at $revision, expected $EXPECTED_REVISION." >&2
    echo "Pass --allow-other-revision to measure a different revision." >&2
    exit 2
fi
echo "boys revision: $revision (dirty paths: $dirty)"
if [ "$dirty" -ne 0 ]; then
    echo "warning: the boys tree has uncommitted changes; the figures belong to the" >&2
    echo "working tree, not to $revision" >&2
fi

mkdir -p "$OUT_DIR" "$BUILD_DIR"
arguments="$OUT_DIR/arguments.csv"
reference="$OUT_DIR/reference.csv"
cells="$OUT_DIR/cells.csv"
environment="$OUT_DIR/environment.txt"
summary="$OUT_DIR/summary.json"

echo "self-testing the analysis (synthetic, no library run)..."
"$PYTHON" "$SCRIPT_DIR/selftest_summarise.py"
"$PYTHON" "$SCRIPT_DIR/selftest_stability.py"

separate="OFF"
if [ "$ROUTE" = "separate" ]; then
    separate="ON"
fi

echo "configuring (Ninja, Release, BOYS_MULADD_SEPARATE=$separate)..."
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBOYS_SOURCE_DIR="$BOYS_SOURCE" \
    -DBOYS_MULADD_SEPARATE="$separate" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

grep -q '^CMAKE_BUILD_TYPE:STRING=Release$' "$BUILD_DIR/CMakeCache.txt" || {
    echo "run_measurement: build type is not Release; refusing a measurement" >&2
    exit 2
}

echo "building measure-interaction..."
cmake --build "$BUILD_DIR" --target measure-interaction
exe="$BUILD_DIR/measure-interaction"
if [ ! -x "$exe" ]; then
    echo "run_measurement: driver not found at $exe" >&2
    exit 2
fi

echo "emitting per-lane arguments..."
"$exe" --emit-arguments "$arguments"

if [ -n "$ARGUMENTS_REFERENCE" ]; then
    if [ ! -f "$ARGUMENTS_REFERENCE" ]; then
        echo "run_measurement: arguments reference '$ARGUMENTS_REFERENCE' does not exist" >&2
        exit 2
    fi
    if ! cmp -s "$arguments" "$ARGUMENTS_REFERENCE"; then
        echo "run_measurement: the emitted narrowed arguments differ from" >&2
        echo "'$ARGUMENTS_REFERENCE'. The configurations are not measuring the same" >&2
        echo "grid, so the comparison would not be like for like. First difference:" >&2
        diff "$ARGUMENTS_REFERENCE" "$arguments" | head -20 >&2 || true
        exit 3
    fi
    echo "arguments identical to '$ARGUMENTS_REFERENCE'"
fi

cp -- "$REFERENCE" "$reference"
echo "using reference '$REFERENCE'"

echo "canary gate (quiet machine required)..."
if ! "$exe" --canary-only; then
    echo "run_measurement: canary gate is wide; the machine is not quiet. Wait for" >&2
    echo "other work to finish and rerun. If the guest is idle, the polluter is on" >&2
    echo "the Windows side: sample the host's process deltas." >&2
    exit 1
fi

echo "measuring (alone)..."
"$exe" --arguments "$arguments" --reference "$reference" \
    --out "$cells" --env "$environment"

route_in_force=$(grep -m1 '^muladd_fp64=' "$environment" | cut -d= -f2)
if [ "$route_in_force" != "$ROUTE" ]; then
    echo "run_measurement: this run asked for the $ROUTE route but the build reports" >&2
    echo "'$route_in_force' in force; the stability premise is not met" >&2
    exit 1
fi

# The repository holding this paper may be a Windows worktree whose gitdir is a
# Windows path that WSL git cannot resolve. The caller can pass the revision and
# dirty count read on the host; otherwise try here and record honestly.
if [ -z "$PAPERS_REVISION" ]; then
    PAPERS_REVISION=$(git -c "safe.directory=$REPO_ROOT" -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)
fi
if [ -z "$PAPERS_REVISION" ]; then
    PAPERS_REVISION="unavailable-in-wsl"
fi
if [ -z "$PAPERS_DIRTY" ]; then
    if dirty_output=$(git -c "safe.directory=$REPO_ROOT" -C "$REPO_ROOT" status --porcelain 2>/dev/null); then
        PAPERS_DIRTY=$(printf '%s' "$dirty_output" | wc -l)
    else
        PAPERS_DIRTY="unavailable-in-wsl"
    fi
fi
cmake_version=$(cmake --version | head -1)
source_date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)

{
    echo ""
    echo "papers_revision=$PAPERS_REVISION"
    echo "papers_dirty_paths=$PAPERS_DIRTY"
    echo "boys_revision=$revision"
    echo "boys_dirty_paths=$dirty"
    echo "target=gcc"
    echo "muladd_route=$ROUTE"
    echo "stability_configuration=gcc-$ROUTE"
    echo "cmake_generator=Ninja"
    echo "cmake_build_type=Release"
    echo "cmake_version=$cmake_version"
    echo "python=$PYTHON"
    echo "source_date_utc=$source_date_utc"
    echo "machine_rule=quiet machine, alone; no build, lane, generator or other timing run concurrently"
    echo "canary_policy=hint-only (owner ruling 2026-10-20; see AMENDMENT-2026-10-20.md)"
    echo "amendment=AMENDMENT-2026-10-20.md"
    echo "coordinator_signal=received"
    echo "canary_read_by=driver (canary: pre-measurement gate, post-measurement reading; see canary_* lines)"
    echo "manifest=manifest.sha256"
} >> "$environment"

if [ -f "$BUILD_DIR/compile_commands.json" ]; then
    cp -- "$BUILD_DIR/compile_commands.json" "$OUT_DIR/compile_commands.json"
fi

echo "summarising..."
"$PYTHON" "$SCRIPT_DIR/summarise_interaction.py" --cells "$cells" --out "$summary"

{
    printf '%s  %s\n' "$(sha256sum "$exe" | cut -d' ' -f1)" "$(basename "$exe")"
    for name in arguments.csv reference.csv cells.csv summary.json; do
        printf '%s  %s\n' "$(sha256sum "$OUT_DIR/$name" | cut -d' ' -f1)" "$name"
    done
    if [ -f "$OUT_DIR/compile_commands.json" ]; then
        printf '%s  %s\n' \
            "$(sha256sum "$OUT_DIR/compile_commands.json" | cut -d' ' -f1)" \
            "compile_commands.json"
    fi
} > "$OUT_DIR/manifest.sha256"

# environment.txt records the manifest's own SHA-256 and is not listed in the
# manifest: a file cannot contain the digest of a manifest that hashes it.
# Verify environment -> manifest_sha256 -> manifest.sha256 -> the files it lists.
printf 'manifest_sha256=%s\n' \
    "$(sha256sum "$OUT_DIR/manifest.sha256" | cut -d' ' -f1)" >> "$environment"

echo "results: $OUT_DIR"
echo "manifest: $OUT_DIR/manifest.sha256"
