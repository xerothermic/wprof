#!/bin/bash
# Refresh PGO profile data for wprof.
# Builds run as current user, only the profiling run uses sudo.
# Usage: ./pgo-refresh.sh [wprof args...]
# Example: ./pgo-refresh.sh -d2000 -f cuda -fpy-torch=nvidia-smi --stat
set -e

if [ "$(id -u)" -eq 0 ]; then
	echo "ERROR: Don't run this script as root/sudo."
	echo "Usage: ./pgo-refresh.sh [wprof args...]"
	echo "The script will sudo only for the profiling run."
	exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PGO_DIR="$SCRIPT_DIR/.output/pgo-profiles"
NPROC="$(nproc)"

if [ $# -eq 0 ]; then
	WPROF_ARGS="-d2000 -f cuda -fpy-torch=nvidia-smi --stat"
else
	WPROF_ARGS="$@"
fi

echo "=== Step 1: Build instrumented binary ==="
make -C "$SCRIPT_DIR" clean && make -C "$SCRIPT_DIR" CC=clang THINLTO=1 PGO_GEN=1 RELEASE=1 -j"$NPROC"

echo ""
echo "=== Step 2: Clear old profiles ==="
rm -f "$PGO_DIR"/*.profraw "$PGO_DIR"/default.profdata

echo ""
echo "=== Step 3: Run workload to collect profiles ==="
echo "Running: sudo $SCRIPT_DIR/wprof $WPROF_ARGS"
sudo "$SCRIPT_DIR/wprof" $WPROF_ARGS || true
# Fix ownership of profraw files written by root
sudo chown "$(id -u):$(id -g)" "$PGO_DIR"/*.profraw 2>/dev/null || true

echo ""
echo "=== Step 4: Merge profiles ==="
PROFRAW_COUNT=$(ls "$PGO_DIR"/*.profraw 2>/dev/null | wc -l)
if [ "$PROFRAW_COUNT" -eq 0 ]; then
	echo "ERROR: No .profraw files generated in $PGO_DIR"
	exit 1
fi
llvm-profdata merge "$PGO_DIR"/*.profraw -o "$PGO_DIR/default.profdata"
echo "Merged $PROFRAW_COUNT profile(s) into default.profdata"
llvm-profdata show "$PGO_DIR/default.profdata" | head -6

echo ""
echo "=== Step 5: Rebuild with PGO ==="
make -C "$SCRIPT_DIR" clean && make -C "$SCRIPT_DIR" CC=clang THINLTO=1 PGO_USE=1 RELEASE=1 -j"$NPROC"

echo ""
echo "=== Done! PGO-optimized binary at $SCRIPT_DIR/wprof ==="
