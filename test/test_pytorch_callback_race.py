"""Repro for the in-flight RecordFunction cached-pointer race.

What this exposes
-----------------
PyTorch's at::RecordFunctionCallback registration model snapshots the
registered start_/end_ function pointers into each RecordFunction
instance's step_callbacks_ member at construction time
(getStepCallbacks(scope), record_function.h:547,500). removeCallback
synchronously erases the entry from PyTorch's global vector but does NOT
visit existing in-flight RecordFunction instances to clear their cached
pointers.

When wprof retracts (rf_unregister calls removeCallback, then dlclose
unmaps libwprofinj.so), any RecordFunction whose `before()` ran while
wprof was active still holds a cached rf_end_cb pointer pointing into
libwprofinj.so. When that RecordFunction's `~RecordFunction` (or
`__exit__` for a `torch.profiler.record_function` block) runs, end()
dispatches the cached pointer into now-unmapped memory and SIGSEGVs.

Workload
--------
Per process (main + forkserver + 2 workers):
  - 4 burst threads:   tight loop of small ops (constant churn).
  - 16 long-hold threads: continuously enter `record_function` blocks of
    HOLD_SECONDS each, with no pause between holds. With 16 such threads
    holding 2-second blocks, several blocks always have a cached
    end_cb pointer at any instant. Wprof retract that overlaps any
    such block exits crashes when the block exits.

Pair with test_pytorch_callback_race.sh which drives wprof retract
cycles and reports the crash rate.
"""
import os
import sys
import time
import threading
import multiprocessing as mp


HOLD_SECONDS = 2.0


def burst_thread(name, stop_evt):
    import torch
    while not stop_evt.is_set():
        a = torch.randn(64, 64)
        b = torch.randn(64, 64)
        c = a @ b
        _ = c.relu().sum()


def long_hold_thread(name, stop_evt):
    """Continuously enter record_function blocks, no pause between."""
    import torch
    n = 0
    while not stop_evt.is_set():
        with torch.profiler.record_function(f"{name}_long_{n}"):
            t_end = time.monotonic() + HOLD_SECONDS
            while time.monotonic() < t_end and not stop_evt.is_set():
                a = torch.randn(128, 128)
                b = torch.randn(128, 128)
                _ = (a @ b).sum()
        n += 1


def run_workload(role):
    print(f"[{role}] PID={os.getpid()} starting workload", flush=True)
    stop_evt = threading.Event()
    threads = []
    for i in range(4):
        threads.append(threading.Thread(
            target=burst_thread,
            args=(f"{role}-burst{i}", stop_evt),
            daemon=True))
    for i in range(16):
        threads.append(threading.Thread(
            target=long_hold_thread,
            args=(f"{role}-long{i}", stop_evt),
            daemon=True))
    for t in threads:
        t.start()

    n = 0
    while True:
        n += 1
        if n % 30 == 0:
            print(f"[{role}] PID={os.getpid()} alive {n}s", flush=True)
        time.sleep(1)


def worker_loop(name):
    run_workload(name)


if __name__ == "__main__":
    import torch
    torch.zeros(1, device='cuda')
    print(f"[main]       PID={os.getpid()}", flush=True)

    # Make _preload_callback_race importable for the forkserver child.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    ctx = mp.get_context('forkserver')
    ctx.set_forkserver_preload(['_preload_callback_race'])

    procs = [ctx.Process(target=worker_loop, args=(f'w{i}',)) for i in range(2)]
    for p in procs:
        p.start()

    fs_pid = mp.forkserver._forkserver._forkserver_pid
    print(f"[main]       worker PIDs={[p.pid for p in procs]}", flush=True)
    print(f"[forkserver] PID={fs_pid}", flush=True)
    # Marker line read by test_pytorch_callback_race.sh:
    print(f"PIDS:{os.getpid()},{fs_pid},{procs[0].pid},{procs[1].pid}",
          flush=True)

    run_workload("main")
