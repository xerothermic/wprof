#!/bin/bash
# Convenience launcher for mlp_bench.py.
# Picks the host-specific xlformers conda python.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WPROF="$SCRIPT_DIR/../src/wprof"
BENCH_PY="$SCRIPT_DIR/mlp_bench.py"
COMPARE_PY="$SCRIPT_DIR/compare_mlp.py"
SIDECAR="/tmp/mlp_bench_self.ndjson"
READY="/tmp/mlp_bench.ready"

if [ "$(uname -m)" = "aarch64" ]; then
    PYTHON="$HOME/xlformer_baseline/conda/bin/python"
else
    PYTHON="$HOME/xlformers_msl_rl_conda/conda/bin/python"
fi
if [ ! -x "$PYTHON" ]; then
    PYTHON=$(command -v python3)
    echo "WARN: conda python not found; falling back to $PYTHON" >&2
fi
echo "Using Python: $PYTHON"

rm -f "$READY" "$SIDECAR"
"$PYTHON" "$BENCH_PY" &
BENCH_PID=$!
echo "Launched mlp_bench.py with PID $BENCH_PID"

for _ in $(seq 1 50); do
    [ -e "$READY" ] && break
    if ! kill -0 "$BENCH_PID" 2>/dev/null; then
        echo "bench process died before becoming ready" >&2
        exit 1
    fi
    sleep 0.1
done

cat <<EOF

Bench is running. Now run wprof + compare, e.g.:

    sudo $WPROF -d1000 -f py-trace=$BENCH_PID -J /tmp/mlp_v1.json
    python3 $COMPARE_PY /tmp/mlp_v1.json

When done:  kill $BENCH_PID

EOF
