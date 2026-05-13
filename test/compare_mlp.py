#!/usr/bin/env python3
"""Compare wprof py-trace `training_step` durations vs Python self-times
recorded by test/mlp_bench.py.

Usage:
    python3 test/compare_mlp.py <wprof.json> [<sidecar.ndjson>]

Sidecar default: /tmp/mlp_bench_self.ndjson.

Pairs the LAST N sidecar entries (where N = number of wprof training_step
events) with the wprof events positionally. The bench is uniform-throughput,
so duration-based anchoring is unreliable (every step matches every other
step within tolerance) — tail pairing is more robust as long as you run the
compare promptly after wprof finishes.
"""

import collections
import json
import sys

NAME = "training_step"


def parse_wprof(path):
    """Return list of (entry_ts_s, exit_ts_s, tid) for each training_step pair."""
    pairs = []
    with open(path) as f:
        f.readline()  # header
        per_tid = collections.defaultdict(list)
        for line in f:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = ev.get("t")
            if t == "pytrace_entry" and ev.get("name") == NAME:
                per_tid[ev["task"]["tid"]].append(ev["ts"])
            elif t == "pytrace_exit" and ev.get("name") == NAME:
                tid = ev["task"]["tid"]
                stk = per_tid.get(tid)
                if not stk:
                    continue
                pairs.append((stk.pop(), ev["ts"], tid))
    pairs.sort(key=lambda x: x[0])
    return pairs


def parse_sidecar(path):
    """Return list of (t0_mono, t1_mono, t0_real, t1_real, tid)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append((e["t0_mono"], e["t1_mono"], e["t0_real"], e["t1_real"], e["tid"]))
    rows.sort(key=lambda x: x[0])
    return rows


def stats_ns(values):
    if not values:
        return (0, 0, 0, 0, 0)
    vs = sorted(values)
    n = len(vs)
    return (sum(vs) / n, vs[n // 2], vs[min(n - 1, int(n * 0.95))],
            vs[min(n - 1, int(n * 0.99))], vs[-1])


def fmt_ns(v):
    av = abs(v)
    if av >= 1_000_000:
        return f"{v / 1_000_000:8.3f}ms"
    if av >= 1_000:
        return f"{v / 1_000:8.2f}us"
    return f"{v:8.0f}ns"


def main(argv):
    if len(argv) < 2 or len(argv) > 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    wprof_path = argv[1]
    sidecar_path = argv[2] if len(argv) > 2 else "/tmp/mlp_bench_self.ndjson"

    wprof = parse_wprof(wprof_path)
    sidecar = parse_sidecar(sidecar_path)

    print(f"wprof:    {wprof_path}  (training_step events={len(wprof)})")
    print(f"sidecar:  {sidecar_path}  (entries={len(sidecar)})")

    if not wprof:
        print("No training_step events in wprof JSON.", file=sys.stderr)
        sys.exit(1)
    if not sidecar:
        print("No entries in sidecar.", file=sys.stderr)
        sys.exit(1)

    if len(sidecar) < len(wprof):
        print(f"\nERROR: sidecar has fewer entries ({len(sidecar)}) than wprof events "
              f"({len(wprof)}) — bench may not have caught up yet.", file=sys.stderr)
        sys.exit(1)
    n = len(wprof)
    tail = sidecar[-n:]
    print(f"pairing wprof[0:{n}] with sidecar[{len(sidecar) - n}:{len(sidecar)}] (tail)")
    print()

    wprof_durs_ns = []
    py_durs_ns = []
    real_skews_ns = []
    for k in range(n):
        wt0, wt1, _ = wprof[k]
        st0, st1, sr0, sr1, _ = tail[k]
        wprof_durs_ns.append(int((wt1 - wt0) * 1e9))
        py_durs_ns.append(st1 - st0)
        real_skews_ns.append(int((sr1 - sr0) * 1e9) - (st1 - st0))

    overheads = [w - p for w, p in zip(wprof_durs_ns, py_durs_ns)]

    w_mean, w_med, w_p95, w_p99, w_max = stats_ns(wprof_durs_ns)
    p_mean, p_med, p_p95, p_p99, p_max = stats_ns(py_durs_ns)
    o_mean, o_med, o_p95, o_p99, o_max = stats_ns(overheads)
    sk_mean = sum(real_skews_ns) / len(real_skews_ns)
    pct = (sum(overheads) / sum(py_durs_ns)) * 100

    print(f"paired: {n} training_step calls\n")
    print(f"            mean       median       p95        p99       max")
    print(f"  python: {fmt_ns(p_mean)} {fmt_ns(p_med)} {fmt_ns(p_p95)} {fmt_ns(p_p99)} {fmt_ns(p_max)}")
    print(f"  wprof : {fmt_ns(w_mean)} {fmt_ns(w_med)} {fmt_ns(w_p95)} {fmt_ns(w_p99)} {fmt_ns(w_max)}")
    print(f"  ovh   : {fmt_ns(o_mean)} {fmt_ns(o_med)} {fmt_ns(o_p95)} {fmt_ns(o_p99)} {fmt_ns(o_max)}")
    print(f"  real_skew (mean): {fmt_ns(sk_mean)}")
    print()
    print(f"sum python: {sum(py_durs_ns)/1e9:.6f}s")
    print(f"sum wprof:  {sum(wprof_durs_ns)/1e9:.6f}s")
    print(f"sum ovh:    {sum(overheads)/1e9:.6f}s  ({pct:+.2f}% of work)")


if __name__ == "__main__":
    main(sys.argv)
