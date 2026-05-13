#!/bin/bash
# Convenience launcher for bench_pytrace.py.
# Picks an fbcode-platform Python (preferred) or system python3, launches the
# benchmark in the background, waits for it to be ready, and prints the wprof
# command(s) you can copy-paste.
#
# The benchmark runs forever; capture as many wprof traces as you want, then
# run test/compare_pytrace.py against each.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WPROF="$SCRIPT_DIR/../src/wprof"
BENCH_PY="$SCRIPT_DIR/bench_pytrace.py"
COMPARE_PY="$SCRIPT_DIR/compare_pytrace.py"
SIDECAR="/tmp/bench_pytrace_self.ndjson"
READY="/tmp/bench_pytrace.ready"

PYTHON=python3
if [ "$(uname -m)" = "aarch64" ]; then
    PLATFORM_PYTHON=/usr/local/fbcode/platform010-aarch64/bin/python3.12
else
    PLATFORM_PYTHON=/usr/local/fbcode/platform010/bin/python3.12
fi
for candidate in "$PLATFORM_PYTHON" python3; do
    if command -v "$candidate" &>/dev/null; then
        if readelf -s "$(readlink -f "$(command -v "$candidate")")" 2>/dev/null | grep -q 'PyEval_SetProfile'; then
            PYTHON="$candidate"
            break
        fi
    fi
done
echo "Using Python: $PYTHON ($(readlink -f "$(command -v "$PYTHON")"))"

rm -f "$READY" "$SIDECAR"
"$PYTHON" "$BENCH_PY" &
BENCH_PID=$!
echo "Launched bench_pytrace.py with PID $BENCH_PID"

# Wait until bench has finished setup (or the process died).
for _ in $(seq 1 50); do
    [ -e "$READY" ] && break
    if ! kill -0 "$BENCH_PID" 2>/dev/null; then
        echo "bench process died before becoming ready" >&2
        exit 1
    fi
    sleep 0.1
done

cat <<EOF

Bench is running. Now in another shell run wprof one or more times, e.g.:

    sudo $WPROF -d1000 -f py-trace=$BENCH_PID -J /tmp/v1.json
    python3 $COMPARE_PY /tmp/v1.json

    sudo $WPROF -d1000 -f py-trace=$BENCH_PID -J /tmp/v2.json
    python3 $COMPARE_PY /tmp/v2.json

When done:  kill $BENCH_PID

EOF
