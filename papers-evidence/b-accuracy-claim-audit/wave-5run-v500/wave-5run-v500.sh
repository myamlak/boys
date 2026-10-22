#!/usr/bin/env bash
# The one command the quiet window runs: the v5.0.0 five-run wave, its own
# canary bracketed around and between the repetitions, then the pooled read.
#
# Usage: bash wave-5run-v500.sh [reps]      (default 5)
#
# Do not launch this outside the coordinator's quiet window. The repeat
# script claims _boys-build/timing.lock and refuses to start while a compiler,
# cmake or make is alive; the canary it writes is the window's own record.
set -u
set -o pipefail

B=/mnt/c/Users/myaml/dev/_boys-build
D=$B/wave-5run-v500
OUT=$D/repeat
REPS=${1:-5}
LOG=$D/repeat-5run-v500.log

mkdir -p "$OUT"

{
    echo "[$(date -u +%FT%TZ)] wave-5run-v500 wrapper start; reps=$REPS"
    bash "$D/repeat-5run-v500.sh" "$REPS" "$OUT" 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    echo "[$(date -u +%FT%TZ)] repeat script rc=$rc"
    if [ "$rc" -ne 0 ]; then
        echo "[$(date -u +%FT%TZ)] wave aborted; no pooling"
        exit "$rc"
    fi
    python3 "$D/pool-5run-v500.py" "$REPS" "$OUT" 2>&1 | tee -a "$LOG"
    prc=${PIPESTATUS[0]}
    echo "[$(date -u +%FT%TZ)] pool rc=$prc"
    exit "$prc"
} 2>&1 | tee -a "$D/wave-5run-v500.log"
exit "${PIPESTATUS[0]}"
