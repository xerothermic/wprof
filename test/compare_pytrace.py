#!/usr/bin/env python3
"""Compare wprof py-trace durations vs Python self-timings from bench_pytrace.py.

Usage:
    python3 test/compare_pytrace.py <wprof.json> [<sidecar.ndjson>]

Defaults sidecar to /tmp/bench_pytrace_self.ndjson.

Matching strategy:
    The bench keeps running while you take wprof captures, so the sidecar
    contains many more entries than the wprof window. We anchor by treating
    the wprof bench-function call sequence (in time order) as a contiguous
    subsequence of the sidecar's bench-function sequence, and find the offset
    by matching (name, duration) pairs. Once anchored, every wprof call has
    an exact corresponding sidecar entry.

The aggregate report bins calls by `depth` (number of nested calls below this
one in its chain — 0 = leaf, only own-work, no children).
"""

import collections
import json
import re
import sys

NAME_RE = re.compile(r"^f_\d+$")          # bench-generated function name pattern
DUR_TOL_MS = 5.0                          # duration tolerance during anchor search
ANCHOR_VALIDATE = 30                      # # of consecutive matches to confirm anchor


def parse_wprof_bench(path):
    """Return (header, [(name, entry_ts_s, exit_ts_s, tid)]) restricted to bench funcs."""
    events = []
    with open(path) as f:
        header = json.loads(f.readline())
        for _ in range(int(header.get("stack_cnt", 0))):
            f.readline()
        per_tid = collections.defaultdict(list)
        for line in f:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue   # tolerate wprof emitting unescaped control chars in non-bench events
            t = ev.get("t")
            if t == "pytrace_entry":
                per_tid[ev["task"]["tid"]].append((ev["name"], ev["ts"]))
            elif t == "pytrace_exit":
                tid = ev["task"]["tid"]
                stk = per_tid.get(tid)
                if not stk:
                    continue
                name, entry = stk.pop()
                if NAME_RE.match(name):
                    events.append((name, entry, ev["ts"], tid))
    events.sort(key=lambda x: x[1])
    return header, events


def parse_sidecar(path):
    """Return [(name, t0_mono, t1_mono, t0_real, t1_real, tid, depth)] sorted by t0_mono."""
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
            if not NAME_RE.match(e["name"]):
                continue
            rows.append((e["name"], e["t0_mono"], e["t1_mono"],
                         e["t0_real"], e["t1_real"], e["tid"], e.get("depth", -1)))
    rows.sort(key=lambda x: x[1])
    return rows


def find_alignment(wprof, sidecar):
    """Find offset i where sidecar[i:i+len(wprof)] aligns with wprof."""
    if not wprof:
        return -1
    w0_name, w0_entry, w0_exit, _ = wprof[0]
    w0_dur_ms = (w0_exit - w0_entry) * 1000
    candidates = [i for i, row in enumerate(sidecar)
                  if row[0] == w0_name and abs((row[2] - row[1]) / 1e6 - w0_dur_ms) <= DUR_TOL_MS]

    for i in candidates:
        if i + min(ANCHOR_VALIDATE, len(wprof)) > len(sidecar):
            continue
        ok = True
        for j in range(min(ANCHOR_VALIDATE, len(wprof))):
            wname, wentry, wexit, _ = wprof[j]
            sname, st0, st1, *_ = sidecar[i + j]
            if wname != sname:
                ok = False; break
            if abs((wexit - wentry) * 1000 - (st1 - st0) / 1e6) > DUR_TOL_MS:
                ok = False; break
        if ok:
            return i
    return -1


def stats_ns(values):
    if not values:
        return (0, 0, 0, 0)
    vs = sorted(values)
    n = len(vs)
    return (sum(vs) / n, vs[n // 2], vs[min(n - 1, int(n * 0.99))], vs[-1])


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
    sidecar_path = argv[2] if len(argv) > 2 else "/tmp/bench_pytrace_self.ndjson"

    header, wprof = parse_wprof_bench(wprof_path)
    sidecar = parse_sidecar(sidecar_path)

    print(f"wprof:    {wprof_path}  (dur={header.get('dur', 0):.3f}s, bench events={len(wprof)})")
    print(f"sidecar:  {sidecar_path}  (bench entries={len(sidecar)})")

    if not wprof:
        print("No bench-function pytrace events in wprof JSON.", file=sys.stderr)
        sys.exit(1)
    if not sidecar:
        print("No bench-function entries in sidecar.", file=sys.stderr)
        sys.exit(1)

    offset = find_alignment(wprof, sidecar)
    if offset < 0:
        print(f"\nERROR: could not anchor wprof sequence in sidecar within "
              f"{DUR_TOL_MS}ms tol over {ANCHOR_VALIDATE} consecutive events.",
              file=sys.stderr)
        sys.exit(1)
    print(f"anchored at sidecar offset {offset} (out of {len(sidecar)})")
    print()

    # Bucket per-call diffs by depth (0 = leaf).
    by_depth = collections.defaultdict(list)
    n_pairs = min(len(wprof), len(sidecar) - offset)
    for k in range(n_pairs):
        wname, wentry, wexit, _ = wprof[k]
        sname, st0, st1, sr0, sr1, _, depth = sidecar[offset + k]
        if wname != sname:
            print(f"WARN: name mismatch at pair {k}: wprof={wname} sidecar={sname}", file=sys.stderr)
            continue
        wprof_dur_ns = int((wexit - wentry) * 1e9)
        py_self_ns = st1 - st0
        py_real_ns = int((sr1 - sr0) * 1e9)
        by_depth[depth].append((wprof_dur_ns, py_self_ns, py_real_ns - py_self_ns))

    cols = ("depth_below", "count", "wprof_mean", "py_mean", "ovh_mean", "ovh_p99", "ovh_max", "real_skew")
    print(f"{cols[0]:<12} {cols[1]:>5}  {cols[2]:>10} {cols[3]:>10}  {cols[4]:>10} {cols[5]:>10} {cols[6]:>10}  {cols[7]:>10}")
    print("-" * 100)
    grand_w = 0; grand_p = 0; grand_n = 0
    for depth in sorted(by_depth):
        rows = by_depth[depth]
        ws  = [r[0] for r in rows]
        ps  = [r[1] for r in rows]
        ovs = [w - p for w, p in zip(ws, ps)]
        sks = [r[2] for r in rows]
        w_mean, *_ = stats_ns(ws)
        p_mean, *_ = stats_ns(ps)
        o_mean, _, o_p99, o_max = stats_ns(ovs)
        sk_mean = sum(sks) / len(sks)
        label = f"{depth} (leaf)" if depth == 0 else str(depth)
        print(f"{label:<12} {len(rows):>5}  "
              f"{fmt_ns(w_mean):>10} {fmt_ns(p_mean):>10}  "
              f"{fmt_ns(o_mean):>10} {fmt_ns(o_p99):>10} {fmt_ns(o_max):>10}  "
              f"{fmt_ns(sk_mean):>10}")
        grand_w += sum(ws); grand_p += sum(ps); grand_n += len(rows)
    print("-" * 100)

    leaf = by_depth.get(0, [])
    if leaf:
        leaf_ovs = [w - p for w, p, _ in leaf]
        leaf_pct = (sum(leaf_ovs) / sum(p for _, p, _ in leaf)) * 100
        print(f"leaf (depth=0): {len(leaf)} calls, "
              f"mean overhead {fmt_ns(sum(leaf_ovs)/len(leaf))} "
              f"({leaf_pct:+.2f}% of work time)")
    print(f"all paired: {grand_n} calls, sum wprof {grand_w/1e9:.6f}s, "
          f"sum python {grand_p/1e9:.6f}s, sum overhead {(grand_w-grand_p)/1e9:.6f}s")
    print("(NB: 'sum' rows double-count nested calls; use leaf line and per-depth ovh_mean for true per-call overhead.)")


if __name__ == "__main__":
    main(sys.argv)
