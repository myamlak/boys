#!/usr/bin/env bash
# The five-run wave at boys v5.0.0 (482fd33): the same four sweeps and both
# whole streams as repeat-5run.sh, REPS times, against the v5.0.0 instruments,
# bracketed by the far-kernel drift canary.
#
# Usage: bash repeat-5run-v500.sh [reps] [out-dir]
#   default reps 5; default out-dir wave-5run-v500/repeat
#
# The case order is the v4.0.0 wave's order, unchanged: region, order,
# stratify, gate-order, whole uniform, whole molecular. Each repetition is a
# full pass of the same instruments, so the pooled per-cell ratios carry a
# between-pass spread.
#
# Every timing figure is read in the window this script's own canary
# describes. The canary (far-kernel's mode "a" against mode "a2", the same
# computation repeated at the end) is run once before the wave, once before
# every repetition, and once after the last; each invocation prints one delta
# per n = 0,1,2. The deltas are written to canary.log beside the repetition
# files, and pool-5run-v500.py refuses to pool without them. A wide canary
# marks every pooled ratio provisional; it does not silently pass.
set -u

B=/mnt/c/Users/myaml/dev/_boys-build
TREE=/mnt/c/Users/myaml/dev/_boys-v500-482fd33
RUN=/mnt/c/Users/myaml/dev/boys-libint2-run-v500
OUT=${2:-$B/wave-5run-v500/repeat}
LOCK=$B/timing.lock
REPS=${1:-5}
HDR=$TREE/include/boys/boys_coefficients.hpp
CANARY_DIR=$B/far-kernel
CANARY_BIN=$CANARY_DIR/far_kernel_cx2
CANARY_LOG=$OUT/canary.log

say() { printf '%s %s\n' "[$(date -u +%FT%TZ)]" "$*"; }

mkdir -p "$OUT"

for bin in benchmark-far-exp benchmark-gate-exp; do
    [ -x "$RUN/$bin" ] || { say "MISSING instrument: $RUN/$bin"; exit 2; }
    [ "$RUN/$bin" -ot "$HDR" ] && { say "STALE instrument: $RUN/$bin is older than $HDR"; exit 2; }
done
[ -x "$CANARY_BIN" ] || { say "MISSING canary: $CANARY_BIN"; exit 2; }

n=0
while [ -e "$LOCK" ] || pgrep -x cc1plus >/dev/null 2>&1 || pgrep -x cmake >/dev/null 2>&1 ||
      pgrep -x make >/dev/null 2>&1 || pgrep -x gmake >/dev/null 2>&1; do
    n=$((n + 1))
    [ "$n" -gt 240 ] && { say "GATE-TIMEOUT"; exit 2; }
    sleep 15
done

n=0
while ! ( set -C; : > "$LOCK" ) 2>/dev/null; do
    n=$((n + 1))
    [ "$n" -gt 240 ] && { say "CLAIM-TIMEOUT"; exit 2; }
    sleep 15
done
trap 'rm -f "$LOCK"' EXIT
say "lock claimed; reps=$REPS; out=$OUT; loadavg $(cut -d' ' -f1-3 /proc/loadavg)"

# What this run is, in the run's own words. Written before the first canary so
# a failure later still leaves the provenance.
# Memory is read first: awk's own quoting cannot nest inside the double-quoted
# command substitutions below.
MEM_TOTAL_GIB=$(free -b | awk '/^Mem:/{printf "%.2f", $2/1073741824}')
MEM_AVAILABLE_GIB=$(free -b | awk '/^Mem:/{printf "%.2f", $7/1073741824}')
{
    echo "started_utc=$(date -u +%FT%TZ)"
    echo "reps=$REPS"
    echo "out=$OUT"
    echo "host=$(uname -srm)"
    echo "nproc=$(nproc)"
    echo "gxx=$(g++ --version | head -1)"
    echo "tree=$TREE"
    echo "tree_revision=$(git -C "$TREE" rev-parse HEAD 2>&1)"
    echo "tree_tag=$(git -C "$TREE" describe --tags --exact-match HEAD 2>&1)"
    echo "gate_header=$HDR"
    echo "gate_header_sha256=$(sha256sum "$HDR" | cut -d' ' -f1)"
    echo "archive=$B/rebuild-instr-v500-build/libboys.a"
    echo "archive_sha256=$(sha256sum "$B/rebuild-instr-v500-build/libboys.a" | cut -d' ' -f1)"
    echo "benchmark_far_exp_sha256=$(sha256sum "$RUN/benchmark-far-exp" | cut -d' ' -f1)"
    echo "benchmark_gate_exp_sha256=$(sha256sum "$RUN/benchmark-gate-exp" | cut -d' ' -f1)"
    echo "canary_bin=$CANARY_BIN"
    echo "canary_sha256=$(sha256sum "$CANARY_BIN" | cut -d' ' -f1)"
    echo "os_release=$(sed -n 's/^PRETTY_NAME=//p' /etc/os-release | tr -d '\"')"
    echo "mem_total_gib=$MEM_TOTAL_GIB"
    echo "mem_available_gib=$MEM_AVAILABLE_GIB"
    echo "power_plan=$(/mnt/c/Windows/System32/powercfg.exe /getactivescheme 2>&1 | tr -d '\r')"
    echo "loadavg_start=$(cut -d' ' -f1-3 /proc/loadavg)"
} > "$OUT/context.txt"
ps -eo pid,ppid,comm,args --sort=comm > "$OUT/processes-at-start.txt" 2>&1

# canary <tag>: one far-kernel drift reading, three deltas in n = 0,1,2 order.
# The raw output is kept beside the parsed line. A reading that does not parse
# stops the wave: a window whose canary cannot be quoted cannot produce a
# figure the record may use.
canary() {
    local tag=$1
    local raw="$OUT/canary-$tag.raw" rc d words
    ( cd "$CANARY_DIR" && "$CANARY_BIN" --n 0,1,2 --count 262144 ) > "$raw" 2>&1
    rc=$?
    d=$(grep -E 'canary a vs a2:' "$raw" | sed -nE 's/.*delta ([+-][0-9.]+)%.*/\1/p' | tr '\n' ' ' | sed -e 's/^ *//' -e 's/ *$//')
    words=$(printf '%s' "$d" | wc -w)
    if [ "$rc" -ne 0 ] || [ "$words" -ne 3 ]; then
        say "  canary $tag FAILED rc=$rc words=$words raw=$raw"
        printf '%s %s PARSE-ERROR\n' "$(date -u +%FT%TZ)" "$tag" >> "$CANARY_LOG"
        exit 3
    fi
    say "  canary $tag: $d %"
    printf '%s %s %s\n' "$(date -u +%FT%TZ)" "$tag" "$d" >> "$CANARY_LOG"
}

: > "$CANARY_LOG"
say "== canary wave-before"
canary wave-before

cd "$RUN" || exit 1

for r in $(seq 1 "$REPS"); do
    say "== repetition $r start; loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
    say "== canary rep$r-before"
    canary "rep$r-before"
    t0=$(date +%s)
    ./benchmark-far-exp --region-sweep  > "$OUT/rep$r-region.txt"   2>&1; say "  region     rc=$?"
    ./benchmark-far-exp --order-sweep   > "$OUT/rep$r-order.txt"    2>&1; say "  order      rc=$?"
    ./benchmark-far-exp --stratify molecular > "$OUT/rep$r-stratify.txt" 2>&1; say "  stratify   rc=$?"
    ./benchmark-gate-exp --order-sweep  > "$OUT/rep$r-gate-order.txt" 2>&1; say "  gate-order rc=$?"
    ./benchmark-far-exp uniform         > "$OUT/rep$r-whole-uniform.txt"   2>&1; say "  uniform    rc=$?"
    ./benchmark-far-exp molecular       > "$OUT/rep$r-whole-molecular.txt" 2>&1; say "  molecular  rc=$?"
    say "== canary rep$r-after"
    canary "rep$r-after"
    say "== repetition $r end secs=$(( $(date +%s) - t0 )); loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
done

say "== canary wave-after"
canary wave-after
ps -eo pid,ppid,comm,args --sort=comm > "$OUT/processes-at-end.txt" 2>&1
say "loadavg end $(cut -d' ' -f1-3 /proc/loadavg)"
say "== $REPS repetition(s) done; pooling is a separate read of $OUT"
