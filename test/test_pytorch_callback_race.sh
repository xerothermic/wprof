#!/bin/bash
# Repro harness for the in-flight RecordFunction cached-pointer race.
#
# Starts test_pytorch_callback_race.py, runs N wprof inject/retract cycles,
# restarts the workload after each iteration so we get an independent trial,
# and reports per-iteration crash result + final summary.
#
# Usage:
#   bash test_pytorch_callback_race.sh [N_ITERATIONS] [WPROF_BIN]
# Defaults:
#   N_ITERATIONS=10
#   WPROF_BIN=$SCRIPT_DIR/../src/wprof  (falls back to /usr/bin/wprof)
#
# Expected output against an unfixed wprof: high crash count
# (~10/10 in our measurements). Per-crash core dumps land in $PWD/core.*
# with the at::RecordFunction::end -> tryRunCallback signature.
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
N="${1:-10}"
WPROF="${2:-$SCRIPT_DIR/../src/wprof}"
[ -x "$WPROF" ] || WPROF="/usr/bin/wprof"

PYTHON=python3
if [ "$(uname -m)" = "aarch64" ]; then
    PLATFORM_PYTHON=/usr/local/fbcode/platform010-aarch64/bin/python3.12
else
    PLATFORM_PYTHON=/usr/local/fbcode/platform010/bin/python3.12
fi
for candidate in "${PYTORCH_PYTHON:-}" "$PLATFORM_PYTHON" "$HOME/xlformer_baseline/conda/bin/python" python3; do
    if [ -x "$candidate" ] 2>/dev/null && "$candidate" -c 'import torch' 2>/dev/null; then
        PYTHON="$candidate"; break
    fi
done

WORKLOAD="$SCRIPT_DIR/test_pytorch_callback_race.py"
LOG=/tmp/test_pytorch_callback_race.repro.log

# Time after retract during which a cached-pointer crash can fire is bounded
# by HOLD_SECONDS in the .py (longest outstanding RecordFunction). Wait
# longer than that so we observe any crash that's going to happen.
HOLD_SECONDS=2
POST_RETRACT_WAIT=$((HOLD_SECONDS + 2))

cleanup_all() {
    pkill -9 -f "python $WORKLOAD" 2>/dev/null
    pkill -9 -f "from multiprocessing.forkserver" 2>/dev/null
    pkill -9 -f "_preload_callback_race" 2>/dev/null
    sleep 2
}

start_workload() {
    local logfile="$1"
    rm -f "$logfile"
    "$PYTHON" "$WORKLOAD" >"$logfile" 2>&1 &
    local wait_for=20
    while ((wait_for-- > 0)); do
        grep -q "^PIDS:" "$logfile" 2>/dev/null && break
        sleep 0.5
    done
    if ! grep -q "^PIDS:" "$logfile"; then
        echo "ERROR: workload did not print PIDS line within 10s" >&2
        cat "$logfile" >&2
        return 1
    fi
    sleep 2  # let threads enter blocks before wprof attaches
    grep "^PIDS:" "$logfile" | tail -1
    return 0
}

alive() { kill -0 "$1" 2>/dev/null; }

declare -i crashes=0 ok=0 first_crash_iter=-1

echo "wprof binary:        $WPROF"
echo "python:              $PYTHON"
echo "iterations:          $N"
echo "post-retract wait:   ${POST_RETRACT_WAIT}s"
echo

for ((i=1; i<=N; i++)); do
    cleanup_all
    pids_line="$(start_workload "$LOG")" || break
    pids="${pids_line#PIDS:}"
    IFS=, read -r main_pid fs_pid w0_pid w1_pid <<<"$pids"
    printf "iter %2d  main=%s fs=%s w0=%s w1=%s  " \
        "$i" "$main_pid" "$fs_pid" "$w0_pid" "$w1_pid"

    sudo "$WPROF" -d 1500 \
        -f py-torch="$main_pid" -f py-torch="$fs_pid" \
        >/tmp/test_pytorch_callback_race.wprof.log 2>&1
    wprof_rc=$?

    sleep "$POST_RETRACT_WAIT"

    dead=""
    for p in "$main_pid" "$fs_pid" "$w0_pid" "$w1_pid"; do
        alive "$p" || dead="$dead $p"
    done

    if [[ -n "$dead" ]]; then
        crashes+=1
        (( first_crash_iter < 0 )) && first_crash_iter=$i
        echo "CRASH (dead:$dead, wprof rc=$wprof_rc)"
    else
        ok+=1
        echo "ok (wprof rc=$wprof_rc)"
    fi
done

cleanup_all

echo
echo "==== summary ===="
echo "crashes:           $crashes / $N"
echo "ok:                $ok / $N"
echo "first crash iter:  $first_crash_iter"
