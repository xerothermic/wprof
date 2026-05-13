#!/usr/bin/env python3
"""PyTorch CPU benchmark designed to expose wprof py-trace overhead.

Uses a deep stack of small Linear+ReLU pairs so each forward call invokes
many Python wrappers around relatively cheap matmuls — the regime where
per-callback Python tracing overhead is most visible. Each step self-times
itself with time.perf_counter_ns() and writes a sidecar entry. Compare via
test/compare_mlp.py.

Workflow:
    ~/xlformer_baseline/conda/bin/python test/mlp_bench.py &
    sudo ./src/wprof -d1000 -f py-trace=<PID> -J /tmp/mlp_v1.json
    python3 test/compare_mlp.py /tmp/mlp_v1.json
"""

import json
import os
import sys
import threading
import time

import torch
import torch.nn as nn

# Tune these to control per-step duration. Many small layers + small batch
# is the regime where per-Python-call tracing cost stays visible (heavy
# Conv/Attention layers hide tracing overhead inside multi-ms BLAS work).
BATCH          = 32
INPUT_DIM      = 64
HIDDEN_DIM     = 64
NUM_LAYERS     = 150
NUM_CLASSES    = 10
LR             = 1e-3
GRAD_CLIP      = 1.0
NUM_THREADS    = 8                # cap intra-op threads so per-step time is reproducible

SIDECAR_PATH   = "/tmp/mlp_bench_self.ndjson"
READY_FILE     = "/tmp/mlp_bench.ready"
FLUSH_EVERY    = 100


def make_model():
    layers = [nn.Linear(INPUT_DIM, HIDDEN_DIM), nn.ReLU()]
    for _ in range(NUM_LAYERS - 1):
        layers += [nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU()]
    layers.append(nn.Linear(HIDDEN_DIM, NUM_CLASSES))
    return nn.Sequential(*layers)


def training_step(model, optimizer, criterion, x, y, sidecar, tid, step_idx, write):
    t0_mono = time.perf_counter_ns()
    t0_real = time.time()
    optimizer.zero_grad(set_to_none=True)
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    t1_mono = time.perf_counter_ns()
    t1_real = time.time()
    write(sidecar, step_idx, tid, t0_mono, t1_mono, t0_real, t1_real, loss.item())


def _write_entry(sidecar, step_idx, tid, t0_mono, t1_mono, t0_real, t1_real, loss):
    sidecar.write(json.dumps({
        "name":    "training_step",
        "step":    step_idx,
        "tid":     tid,
        "t0_mono": t0_mono,
        "t1_mono": t1_mono,
        "t0_real": t0_real,
        "t1_real": t1_real,
        "loss":    loss,
    }, separators=(",", ":")) + "\n")


def main():
    torch.set_num_threads(NUM_THREADS)
    device = torch.device("cpu")
    model = make_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    x = torch.randn(BATCH, INPUT_DIM, device=device)
    y = torch.randint(0, NUM_CLASSES, (BATCH,), device=device)

    sidecar = open(SIDECAR_PATH, "w", buffering=1 << 16)
    tid = threading.get_native_id()
    pid = os.getpid()
    print(f"PID: {pid}", flush=True)
    print(f"TID: {tid}", flush=True)
    print(f"sidecar: {SIDECAR_PATH}", flush=True)
    print(f"batch={BATCH} hidden={HIDDEN_DIM} layers={NUM_LAYERS} threads={NUM_THREADS} "
          f"params={sum(p.numel() for p in model.parameters())}", flush=True)
    open(READY_FILE, "w").close()

    step = 0
    try:
        while True:
            training_step(model, optimizer, criterion, x, y, sidecar, tid, step, _write_entry)
            step += 1
            if step % FLUSH_EVERY == 0:
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
        print(f"completed steps: {step}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
