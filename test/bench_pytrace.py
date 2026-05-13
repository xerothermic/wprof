#!/usr/bin/env python3
"""Python tracing accuracy benchmark for wprof's `-f py-trace` feature.

Generates a long stream of nested function calls with configurable diversity
(pool size), depth, and per-call work, recording each call's own start/end
timestamps for diffing against wprof's `-J` JSON output (see
test/compare_pytrace.py).

Knobs are constants below — edit and re-run.

Workflow:
    python3 test/bench_pytrace.py &        # prints `PID: <N>`, runs forever
    sudo ./src/wprof -d1000 -f py-trace=<N> -J /tmp/v1.json
    python3 test/compare_pytrace.py /tmp/v1.json
    # repeat wprof + compare with v2.json, v3.json, ...

Time sources:
    * t0_mono / t1_mono   = time.perf_counter_ns()  (CLOCK_MONOTONIC, matches
                            wprof's internal clock — see src/inj.h:44-51)
    * t0_real / t1_real   = time.time()             (CLOCK_REALTIME)
"""

import json
import os
import random
import sys
import threading
import time

# ---------------------------------------------------------------------------
# Configurable knobs
# ---------------------------------------------------------------------------

POOL_SIZE      = 500                             # total distinct generated functions f_000..f_499
DEPTH_RANGE    = (100, 300)                      # deep chains (mimic PyTorch autograd nesting)
WORK_MS_RANGE  = (0.005, 0.05)                   # 5-50 us per call (mimic small-Linear regime)
ITERATIONS     = None                            # None = run forever, int = stop after N
RNG_SEED       = 0xC0FFEE                        # fixed for reproducibility

SIDECAR_PATH   = "/tmp/bench_pytrace_self.ndjson"
READY_FILE     = "/tmp/bench_pytrace.ready"

FLUSH_EVERY    = 50                              # flush sidecar every N iterations

# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------

# Each generated function has the same body but a distinct co_qualname so
# wprof sees POOL_SIZE different code objects (stresses the per-code cache).
# A call chain is a list of distinct functions; each function does its own
# work, then if the chain has more elements it dispatches to chain[1].
_FUNC_TEMPLATE = """\
def {name}(chain, work, iter_idx, sidecar, tid, write_entry):
    t0_mono = time.perf_counter_ns()
    t0_real = time.time()
    acc = 0
    for i in range(work):
        acc += i * i
    if chain:
        acc += chain[0](chain[1:], work, iter_idx, sidecar, tid, write_entry)
    t1_mono = time.perf_counter_ns()
    t1_real = time.time()
    write_entry(sidecar, "{name}", iter_idx, tid,
                t0_mono, t1_mono, t0_real, t1_real, len(chain))
    return acc
"""

def _build_pool():
    ns = {"time": time}
    for i in range(POOL_SIZE):
        exec(_FUNC_TEMPLATE.format(name=f"f_{i:03d}"), ns)
    return [ns[f"f_{i:03d}"] for i in range(POOL_SIZE)]

def _calibrate_iters_per_ms(target_ns=20_000_000):
    """Return iterations of the inner loop that take ~1ms on this host."""
    iters = 10_000
    while True:
        acc = 0
        t0 = time.perf_counter_ns()
        for i in range(iters):
            acc += i * i
        elapsed = time.perf_counter_ns() - t0
        if elapsed >= target_ns:
            return iters / (elapsed / 1_000_000.0)
        iters *= 2

def _write_entry(sidecar, name, iter_idx, tid, t0_mono, t1_mono, t0_real, t1_real, depth_below):
    sidecar.write(json.dumps({
        "name":    name,
        "iter":    iter_idx,
        "tid":     tid,
        "t0_mono": t0_mono,
        "t1_mono": t1_mono,
        "t0_real": t0_real,
        "t1_real": t1_real,
        "depth":   depth_below,    # number of nested calls below this one (0 = leaf)
    }, separators=(",", ":")) + "\n")

def main():
    sidecar = open(SIDECAR_PATH, "w", buffering=1 << 16)

    pool = _build_pool()
    rng = random.Random(RNG_SEED)
    tid = threading.get_native_id()
    pid = os.getpid()
    iters_per_ms = _calibrate_iters_per_ms()

    print(f"PID: {pid}", flush=True)
    print(f"TID: {tid}", flush=True)
    print(f"sidecar: {SIDECAR_PATH}", flush=True)
    print(f"pool_size={POOL_SIZE} depth_range={DEPTH_RANGE} work_ms={WORK_MS_RANGE} "
          f"iters={ITERATIONS} calibrated_iters_per_ms={iters_per_ms:.0f}", flush=True)

    open(READY_FILE, "w").close()

    i = 0
    try:
        while ITERATIONS is None or i < ITERATIONS:
            d = rng.randint(*DEPTH_RANGE)
            w = int(rng.uniform(*WORK_MS_RANGE) * iters_per_ms)
            chain = rng.sample(pool, d)
            chain[0](chain[1:], w, i, sidecar, tid, _write_entry)
            i += 1
            if i % FLUSH_EVERY == 0:
                sidecar.flush()
    except KeyboardInterrupt:
        pass
    finally:
        sidecar.flush()
        sidecar.close()
        try:
            os.unlink(READY_FILE)
        except FileNotFoundError:
            pass
        print(f"completed iterations: {i}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()
