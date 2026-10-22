#!/bin/bash
# Rebuild the five-run comparator instruments against boys v5.0.0 (482fd33).
#
#   bash /mnt/c/Users/myaml/dev/_boys-build/wave-5run-v500/rebuild-instruments-v500.sh [tree]
#
# This is rebuild-instruments-5run.sh re-pointed at the v5.0.0 tree. The tree
# defaults to a plain shared clone of the release, HEAD 482fd33, described by
# the exact-match tag v5.0.0, in /mnt/c/Users/myaml/dev/_boys-v500-482fd33.
# The comparator source, the flag string, the include order and the archive
# target are the ones the v4.0.0 wave used; only the tree and the output
# directory change, so the two waves differ in the library and in nothing
# else. policy-ab is not rebuilt: the five-run protocol does not run it, and
# its policy names are a v4.0.0-era probe, not a five-run instrument.
#
# What it does, one step at a time:
#   1. configure the tree Release + AVX2 into a FRESH build dir under
#      _boys-build, build the `boys` archive there.
#   2. compile benchmark-far-exp and benchmark-gate-exp in the v5.0.0 run dir
#      against that archive, with the comparator harness's own flags.
#   3. print every sha256, then apply the freshness gate and refuse (exit 2)
#      if an instrument is older than the generated header.
#
# Machine discipline: every compile waits for the reservation lock and for
# cc1plus, cmake, make and gmake to be gone from WSL. The archive builds at
# -j3, the instruments compile one at a time. Nothing inside the tree or the
# instrument sources is edited.
set -u
set -o pipefail

B=/mnt/c/Users/myaml/dev/_boys-build
TREE=${1:-/mnt/c/Users/myaml/dev/_boys-v500-482fd33}
RUN=/mnt/c/Users/myaml/dev/boys-libint2-run-v500
LIBINT2=/mnt/c/Users/myaml/dev/libint-2.13.1/inst/include
BUILD=$B/rebuild-instr-v500-build
BENCH_SRC=/mnt/c/Users/myaml/dev/_boys-build/wave-5run/new/benchmarks/boys_libint2_benchmark.cpp
HDR=$TREE/include/boys/boys_coefficients.hpp
LOCK=$B/timing.lock
LOG=$B/wave-5run-v500/rebuild-instruments-v500.log
WAIT_MAX=${WAIT_MAX:-5400}

# The five-run comparator's own recipe, unchanged from the v4.0.0 wave.
FLAGS="-std=c++23 -O3 -DNDEBUG -mavx2 -mfma -mf16c -ffp-contract=off -DBOYS_SCALAR_CONTRACTS=0 -DBOYS_SIMD_X86=1 -DBoysFp16=1"

mkdir -p "$(dirname "$LOG")"
say() { printf '%s %s\n' "[$(date +%H:%M:%S)]" "$*" | tee -a "$LOG"; }
die() { local rc=$1; shift; say "FAILED: $*"; for b in benchmark-far-exp benchmark-gate-exp; do rm -f "$RUN/$b.new"; done; say "rebuild-instruments-v500: STOPPED (rc=$rc)"; exit "$rc"; }

busy() {
    pgrep -x cc1plus >/dev/null 2>&1 && return 0
    pgrep -x cmake   >/dev/null 2>&1 && return 0
    pgrep -x gmake   >/dev/null 2>&1 && return 0
    pgrep -x make    >/dev/null 2>&1 && return 0
    return 1
}
running() { pgrep -x "$1" 2>/dev/null | wc -l; }

wait_clear() {
    local waited=0 period=15
    while :; do
        if [ ! -f "$LOCK" ] && ! busy; then
            [ "$waited" -gt 0 ] && say "machine clear after ${waited}s"
            return 0
        fi
        [ "$waited" -ge "$WAIT_MAX" ] && return 1
        if [ $((waited % 120)) -eq 0 ]; then
            say "waiting for a clear machine (${waited}s): lock=$([ -f "$LOCK" ] && echo present || echo absent) cc1plus=$(running cc1plus) cmake=$(running cmake) make=$(running make) gmake=$(running gmake)"
        fi
        sleep "$period"; waited=$((waited + period))
    done
}
gate() { wait_clear || die 3 "machine never went clear within ${WAIT_MAX}s"; }

stage() {
    local name=$1; shift
    say "--- $name"
    say "\$ $*"
    local t0 t1 rc
    t0=$(date +%s)
    "$@" 2>&1 | tee -a "$LOG"; rc=$?
    t1=$(date +%s)
    say "--- $name rc=$rc secs=$((t1 - t0))"
    [ "$rc" -eq 0 ] || die "$rc" "$name exited $rc"
}

for p in "$TREE" "$TREE/CMakeLists.txt" "$HDR" "$RUN" "$RUN/shim/qcx/integrals/boys.hpp" "$BENCH_SRC" "$LIBINT2"; do
    [ -e "$p" ] || die 1 "missing: $p"
done

REV=$(git -C "$TREE" rev-parse HEAD 2>&1) || die 1 "cannot read the revision of $TREE"
TAG=$(git -C "$TREE" describe --tags --exact-match HEAD 2>/dev/null || echo "no exact tag")
DIRTY=$(git -C "$TREE" status --porcelain 2>/dev/null | wc -l)
[ "$TAG" = "v5.0.0" ] || die 1 "tree is not at v5.0.0: tag=$TAG"
[ "$DIRTY" -eq 0 ] || die 1 "tree is dirty: $DIRTY path(s)"

say "===================================================================="
say "rebuild-instruments-v500: tree=$TREE"
say "tree revision : $REV  ($TAG)"
say "tree dirty    : $DIRTY path(s)"
say "build dir     : $BUILD  (fresh; the tree's build dirs are never written)"
say "archive out   : $BUILD/libboys.a   instruments out=$RUN"
say "gate header   : $HDR"
say "                $(stat -c '%y  %s bytes' "$HDR")  sha256=$(sha256sum "$HDR" | cut -d' ' -f1)"
say "flags         : $FLAGS"
say "g++           : $(g++ --version | head -1)"
say "cmake         : $(cmake --version | head -1)"
say "===================================================================="

rm -rf "$BUILD" || die 9 "could not remove $BUILD"
mkdir -p "$BUILD" || die 9 "could not create $BUILD"

export CMAKE_BUILD_PARALLEL_LEVEL=3
gate
stage configure cmake -S "$TREE" -B "$BUILD" -G "Unix Makefiles" \
      -DCMAKE_BUILD_TYPE=Release -DBOYS_ISA_AVX2=ON
gate
stage archive-build cmake --build "$BUILD" --target boys -j3

[ -f "$BUILD/libboys.a" ] || die 1 "no archive at $BUILD/libboys.a after the build"
say "archive: $(stat -c '%y  %s bytes' "$BUILD/libboys.a")"
say "archive md5   : $(md5sum "$BUILD/libboys.a" | cut -d' ' -f1)"
say "archive sha256: $(sha256sum "$BUILD/libboys.a" | cut -d' ' -f1)"

cd "$RUN" || die 9 "cannot cd $RUN"
gate
stage compile-far-exp g++ $FLAGS -I./shim -I"$TREE/include" -I"$LIBINT2" "$BENCH_SRC" "$BUILD/libboys.a" -o benchmark-far-exp.new
gate
stage compile-gate-exp g++ $FLAGS -I./shim -I"$TREE/include" -I"$LIBINT2" "$BENCH_SRC" "$BUILD/libboys.a" -o benchmark-gate-exp.new

for b in benchmark-far-exp benchmark-gate-exp; do
    [ -x "./$b.new" ] || die 1 "staged binary missing after a clean compile: $b.new"
done
for b in benchmark-far-exp benchmark-gate-exp; do
    mv -f "./$b.new" "./$b" || die 1 "could not install $b"
done
say "installed: benchmark-far-exp benchmark-gate-exp"

for bin in benchmark-far-exp benchmark-gate-exp; do
    [ -x "./$bin" ] || die 2 "$bin is missing"
    [ "./$bin" -ot "$HDR" ] && die 2 "$bin is older than $HDR"
done

say "===================================================================="
say "instruments rebuilt against the v5.0.0 tree; the freshness gate passes"
say "gate header : $HDR  $(stat -c '%y' "$HDR")"
say "archive     : $BUILD/libboys.a  sha256=$(sha256sum "$BUILD/libboys.a" | cut -d' ' -f1)"
for bin in benchmark-far-exp benchmark-gate-exp; do
    say "$(printf '%-18s' "$bin"): $(stat -c '%y  %s bytes' "./$bin")  sha256=$(sha256sum "./$bin" | cut -d' ' -f1)"
done
say "log         : $LOG"
say "rebuild-instruments-v500: DONE"
say "===================================================================="
exit 0
