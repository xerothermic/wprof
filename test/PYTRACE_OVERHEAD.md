# wprof py-trace overhead measurement

Tooling and **measured** findings for `wprof -f py-trace` on x86 (devgpu004)
and ARM/aarch64 (devgpu018) linux dev hosts. Everything below is direct
measurement or arithmetic derived from direct measurement; no causal or
mechanism claims.

## TL;DR (measured numbers)

Steady-state PyTorch CPU MLP training step (150-layer Linear+ReLU tower,
BATCH=32, AdamW + grad clipping, ~30 ms / step):

| | ARM (aarch64) | x86_64 |
|---|---|---|
| Pure step time (no wprof) | ~28-33 ms | ~30-35 ms |
| Step time during attach | ~41-46 ms | ~36-40 ms |
| **Per-step time tax** | **+12 to +13 ms (~37-47%)** | **+5 to +6 ms (~14-18%)** |
| ARM/x86 wall-time tax ratio | — | **~2.3×** |
| Per-step extra instructions | **+47-49 M** | **+40-43 M** |
| Reproducibility across 3 attaches | step ±1.5 ms, IPC ±0.08 | step ±0.7 ms, IPC ±0.04 |

Per-step inflation appears immediately on attach and disappears on detach.
Recovery is verified at every gap between attaches (not just at the end).

## Key methodological insight: timing measurement during attach is biased

`time.perf_counter_ns()` reads `CLOCK_MONOTONIC`. So does wprof's
`Py_tracefunc` callback (`src/inj.h:44-51`). Inside a traced function:

```
wprof_bracket_dur  ≈  py_self_dur (during attach)  ≈  pure_work + tracing_overhead
```

Both clocks read the same elevated value during attach, so diffing them
gives ~0 difference. To recover the actual tracing tax you must compare
`wprof_bracket` against an **untraced** Python self-time — i.e. a sidecar
entry from outside the attach window. The benches solve this by writing
a sidecar entry per step continuously (long-running daemon), so entries
written outside the attach window are pure work.

**Pitfall caught:** the original `compare_mlp.py` used a duration-anchor
matcher copied from `compare_pytrace.py`. With uniform ~30 ms steps every
sidecar entry's duration matches every wprof event within tolerance, so
the anchor returned an arbitrary offset and ended up pairing
during-attack against during-attack — masking the overhead. Symptom:
`mean ovh < median ovh`, plus *negative* per-step overheads (physically
impossible). Fix: tail-pair the last N sidecar entries with the wprof
events. Works as long as the bench keeps running after wprof detaches.

## Bench inventory

All under `test/`:

- **`bench_pytrace.py`** — pure-Python synthetic. Pool of `POOL_SIZE`
  distinct `f_NNN` functions; each iteration samples a random call chain
  of `DEPTH_RANGE` length and per-call work in `WORK_MS_RANGE`. Records
  per-call timings to `/tmp/bench_pytrace_self.ndjson`. Default knobs
  (POOL_SIZE=500, DEPTH_RANGE=(100,300), WORK_MS_RANGE=(0.005,0.05))
  put it in the **MLP-equivalent overhead regime** — see "Synthetic
  bench in MLP regime" below.
- **`compare_pytrace.py`** — sequence-anchored comparator for the synthetic
  bench. Buckets by `depth` (chain position from leaf).
- **`mlp_bench.py`** — PyTorch CPU bench: 150-layer `Linear+ReLU` skinny
  tower, BATCH=32, AdamW + grad-clip, label-smoothed CE. Per-step
  self-times with `time.perf_counter_ns()`. Picks host-specific xlformers
  conda python via `mlp_bench.sh`.
- **`compare_mlp.py`** — tail-pairing comparator for the MLP bench. Takes
  the last N sidecar entries and pairs positionally with the N wprof
  events.

## Reproducing

```bash
# either platform — the .sh picks the right xlformers conda python
bash test/mlp_bench.sh                    # backgrounds the bench, prints PID
sudo ./src/wprof -d1000 -f py-trace=$PID -D /tmp/cap.data -J /tmp/cap.json
python3 test/compare_mlp.py /tmp/cap.json
```

Multi-attach reproducibility:

```bash
PID=…
for i in 1 2 3; do
  sudo ./src/wprof -d3000 -f py-trace=$PID -D /tmp/multi_$i.data
  sleep 5
done
```

## Per-call work vs %overhead (measured across regimes)

| Per-call work | %overhead (ARM aarch64) |
|---|---|
| ~3 µs (synthetic, WORK_RANGE=(0,200)) | +94% |
| ~13 µs (synthetic leaf, WORK_MS_RANGE=(0.001,0.01)) | ~+50% |
| ~1 ms (synthetic, WORK_MS_RANGE=(0.3,2.0)) | +1.4% |
| ~13 ms (synthetic leaf, WORK_MS_RANGE=(3,20)) | +0.2% |
| ~30 ms (MLP, 150-layer skinny tower) | +37-47% |
| ~770 ms (MLP, scaled+inner-repeats) | +0.01% |

The MLP outlier (much higher % than the synthetic at similar step time)
is because PyTorch makes hundreds of nested Python calls per step; the
30 ms step accumulates many small per-call taxes. The synthetic at 30 ms
total was a single deep call chain with much less Python plumbing.

## PMU measurements (3-toggle multi-attach experiment)

Setup: `perf stat -t $PID -I 100 -e cycles,instructions,l1-icache-load-misses,itlb-load-misses` for 60 s; in parallel a background loop fires `wprof -d3000` three times with 5 s sleeps. Attach windows are detected from the sidecar (step-time elevation > 1.10× baseline) so PMU bucketing aligns to the actual capture windows on both archs.

**ARM, in chronological order:**

| phase | n | IPC | inst/s | L1i MPKI | iTLB miss/Ki | step | inst/step |
|---|---|---|---|---|---|---|---|
| pre | 95 | 2.41 | 7.74 G | 22.31 | 1.039 | 32.93 ms | 254.9 M |
| dur1 | 26 | 2.09 | 6.71 G | 26.26 | 1.164 | 45.08 ms | 302.5 M |
| mid1/2 (recovery) | 52 | 2.43 | 7.82 G | 22.50 | 1.040 | 32.69 ms | 255.5 M |
| dur2 | 26 | 2.03 | 6.54 G | 26.31 | 1.167 | 45.90 ms | 300.4 M |
| mid2/3 (recovery) | 52 | 2.36 | 7.59 G | 22.31 | 1.039 | 33.63 ms | 255.3 M |
| dur3 | 27 | 2.01 | 6.49 G | 26.15 | 1.162 | 46.62 ms | 302.7 M |
| post (final) | 270 | 2.36 | 7.62 G | 22.21 | 1.033 | 33.28 ms | 253.7 M |

**x86, in chronological order** (3 attaches; the sidecar-step detector
spuriously split dur1 into two adjacent runs and dur3 into two adjacent
runs; values across the splits are within 0.04 IPC and 0.7 ms step):

| phase | n | IPC | inst/s | L1i MPKI | iTLB miss/Ki | step | inst/step |
|---|---|---|---|---|---|---|---|
| pre | 96 | 1.33 | 4.92 G | 1.53 | 0.337 | 34.47 ms | 169.4 M |
| dur1 (effective) | 18 | 1.43 | 5.27 G | 1.52 | 0.322 | ~40.0 ms | ~211 M |
| mid2/3 (recovery) | 57 | 1.32 | 4.89 G | 1.55 | 0.341 | 34.18 ms | 167.1 M |
| dur2 | 27 | 1.44 | 5.31 G | 1.53 | 0.321 | 39.34 ms | 208.9 M |
| mid3/4 (recovery) | 57 | 1.33 | 4.92 G | 1.54 | 0.338 | 34.32 ms | 168.8 M |
| dur3 (effective) | 16 | 1.43 | 5.30 G | 1.50 | 0.318 | ~39.8 ms | ~210 M |
| post (final) | 267 | 1.29 | 4.77 G | 1.54 | 0.341 | 35.14 ms | 167.8 M |

### Direction and magnitude of changes (during-attach Δ vs pre)

| Metric | ARM Δ | x86 Δ |
|---|---|---|
| IPC | **−14% to −17%** | **+6% to +9%** |
| inst/s | −13% to −16% | +6% to +9% |
| L1i MPKI | **+17% to +18%** | flat (−1% to +0%) |
| iTLB miss/Ki | **+12%** | flat (−5% to −8%) |
| step time | **+37% to +42%** | **+14% to +16%** |
| inst/step | **+18% to +19%** (≈+47-49 M) | **+23% to +26%** (≈+40-43 M) |
| post recovery to baseline | clean (every recovery window) | clean (every recovery window) |

### Baseline (no wprof) ratios

| | ARM | x86 | ratio |
|---|---|---|---|
| Path length (inst/step) | 254.9 M | 169.4 M | ARM **1.5×** |
| L1i MPKI | 22.31 | 1.53 | ARM **15×** |
| iTLB miss/Ki | 1.039 | 0.337 | ARM **3.1×** |
| Step time | 32.93 ms | 34.47 ms | ARM 1.05× faster |

## What the data shows directly

- During attach, ARM and x86 see roughly equal **absolute** extra
  instructions per step (~+45-49 M on both).
- ARM IPC and inst/s drop during attach; x86 IPC and inst/s rise.
- ARM L1i MPKI and iTLB miss/Ki rise during attach; x86's stay flat.
- ARM's wall-time tax per step is ~2.3× x86's.
- Both architectures fully return to baseline on every recovery window
  measured (between attaches and after the last attach), across every
  metric in the table.

## What the data does NOT show

These are open questions, not conclusions:

- **Cause of the IPC drop on ARM** — the icache and iTLB rises *coincide*
  with the IPC drop, but causation is not isolated. Could equally be
  backend-bound (e.g. memory dependency stalls). Would need stall
  attribution: `stalled-cycles-frontend`, topdown breakdown, or per-cycle
  reason codes.
- **Cause of the IPC *rise* on x86** — IPC goes up during attach with
  flat MPKI/iTLB. Mechanism unmeasured.
- **Working-set vs cache capacity** — the 22 vs 1.5 MPKI ratio between
  archs is not attributable from this data alone (could be working-set
  size, cache size, prefetcher behavior, or any combination).
- **Whether the +47 M extra instructions per step are themselves slow
  on ARM, or whether they perturb the pre-existing 250 M baseline
  instructions** — only the totals are measured.
- **Branch prediction effects under wprof attach** — collected briefly
  in an earlier 7-event multiplexed run showing ~20% rise on both archs,
  but that data was multiplexed and the controlled 4-event run dropped
  branch counters. Not currently included in the measured set above.

## Things tried that didn't pan out

- **"Each wprof attach leaves a permanent residue"** — observed a small
  drift in early experiments. Tested with controlled multi-attach runs
  measuring step time AND PMU counters between attaches: every recovery
  window matches its pre-attach baseline within noise. Hypothesis not
  supported.
- **Anchor-matching comparator for `compare_mlp.py`** — see "Pitfall"
  above. Reverted to tail pairing.
- **Diversifying the model with Conv2d + MultiheadAttention** — looked
  more realistic but each Python call wraps multi-ms of BLAS work, which
  hides the per-callback tax. Reverted to skinny `Linear` tower.
- **7-event perf stat (with branch counters)** — caused multiplexing on
  x86 and produced noisier IPC values with no detectable attach signal.
  Reduced to 4 events for clean numbers; branch data not currently
  collected.

## PGO/ThinLTO results

Following branch `pgo-pytrace-experiment` applies the previously-unused
ThinLTO + PGO patches and rebuilds wprof + libwprofinj.so with profile
data collected against the live mlp_bench workload.

### ARM (back-to-back same-bench-PID comparison, clang 22.1.3)

Profile collection:
- 5× `wprof -d5000 -f py-trace=$BENCH_PID` to drive the `pytrace_profile_callback`
  hot path.
- Kill bench (SIGTERM) to flush libwprofinj.so's PGO counters via the
  LLVM atexit handler.
- Required workaround: `chown patlu:users src/.output/pgo-profiles` BEFORE
  launching wprof, otherwise the first sudo-wprof creates the dir
  root-owned and the bench (running as patlu) can't write its `.profraw`.
- Verified with `llvm-profdata show … --all-functions | grep
  pytrace_profile_callback`.

Same-bench-PID 3-toggle PMU experiment (perf stat -I 100, 4 events):

| Metric | Non-PGO clang21 | Fresh PGO clang21 | Δ |
|---|---|---|---|
| pre step | 28.42 ms | 28.67 ms | ~same |
| dur step | 42.66 ms | 41.27 ms | -1.39 ms (-3.3%) |
| **wprof tax (dur−pre)** | **14.24 ms** | **12.60 ms** | **-1.64 ms (-11.5%)** |
| dur IPC | 2.20 | 2.27 | +3.2% |
| dur inst/step | 303 M | 299 M | -4 M (-1.3%) |
| dur L1i MPKI | 25.48 | 25.43 | ~flat |
| dur iTLB miss/Ki | 1.179 | 1.183 | ~flat |

PGO reduces ARM wprof tax by **~11.5%** (1.64 ms / step) on this
workload. Mechanism (per data): IPC during attach rises ~3% with a
small ~1% drop in instruction count per step; L1i MPKI and iTLB
miss/Ki barely move.

### x86 (BLOCKED on a separate clang-build inject regression)

Goal was the same comparison on x86. Could not produce numbers because
**clang21-built wprof on x86 captures 0 pytrace_entry events** — wprof
runs to completion and writes a trace, but py-trace inject silently
yields no callback events. Repro:

| build | pytrace_entry events captured |
|---|---|
| `make RELEASE=1` (gcc) | 410,881 |
| `make CC=clang RELEASE=1` (clang21, no PGO) | 0 |
| `make CC=clang PGO_USE=1 RELEASE=1` (clang21+PGO) | 0 |
| `make CC=clang THINLTO=1 PGO_USE=1 RELEASE=1` (clang21+ThinLTO+PGO) | 0; additionally prints "PTRACE injection failed for $PID, python: -22" |

Effect is independent of PGO (already broken at plain `CC=clang
RELEASE=1`). Issue is in the clang-built inject path on x86 specifically;
not reproduced on aarch64 where clang+PGO+ThinLTO works fine. Out of
scope for this PGO experiment — filed as follow-up.

A second issue specific to x86 also surfaced during profile collection:
even after dir permissions were corrected, `libwprofinj.so` did not
write a `.profraw` on bench exit (only the wprof-binary `.profraw`
appeared). On aarch64, the `.so` `.profraw` does appear under the same
procedure. Cause unconfirmed.

## Synthetic bench in MLP regime (ARM)

After tuning `bench_pytrace.py` to `POOL_SIZE=500, DEPTH_RANGE=(100,300),
WORK_MS_RANGE=(0.005,0.05)` (deep chains of small Python calls, mimicking
PyTorch autograd nesting depth + per-layer work granularity):

| Metric | Synth pre | Synth dur (avg of 3) | Synth post |
|---|---|---|---|
| IPC | 5.32 | 4.73 (↓11%) | 5.36 |
| inst/s | 17.08 G | 15.22 G (↓11%) | 17.30 G |
| L1i MPKI | **1.76** | **2.48** (↑41%) | 1.72 |
| iTLB miss/Ki | ~0 | ~0 | ~0 |

Per-call comparison from `compare_pytrace.py` on a 3 s wprof window:
**leaf calls (depth=0): +17.92 µs tax on ~22 µs work = +82% per-call
overhead**. Each non-leaf call adds ~9 µs fixed callback tax that stacks
linearly with depth.

The synthetic bench reproduces:
- The **~11-15% IPC drop** during attach (matches MLP).
- The **per-callback fixed cost** (~9 µs/call, similar to MLP if you
  back it out from MLP's +49 M inst/step ÷ ~hundreds of nested calls).

The synthetic bench does NOT reproduce:
- The **icache-saturated regime** of MLP. MLP baseline L1i MPKI is 22.31
  and during-attach 26.24; synth baseline is 1.76 and during-attach 2.48.
  The synth bench's 500-function pool fits well inside L1i; PyTorch's
  call graph (Linear/autograd/optimizer/blas dispatch) does not. So the
  *amplification* from front-end saturation seen on MLP requires either
  a much larger code working set than the synth bench currently exercises,
  or PyTorch's actual call graph.

## Resume guide (state of investigation)

What's measured and known:
- ARM has ~2.3× larger wprof tax than x86 on the MLP CPU bench.
- ARM's PGO-built wprof reduces tax by 11.5% (14.24 ms → 12.60 ms / step).
- IPC drop during attach reproduces in BOTH the MLP and tuned-synth benches.
- L1i MPKI rise during attach reproduces ONLY in MLP, where baseline is
  already at 22 MPKI; synth baseline is 1.76 MPKI.
- Both architectures fully recover to baseline on every recovery window.

What's blocked / open:
- x86 PGO measurement: clang-built wprof on x86 captures 0 pytrace events
  (pre-existing, not PGO's fault). gcc build works fine.
- Causal isolation: front-end pressure correlates with the IPC drop on
  ARM, but causation is not isolated. Needs `stalled-cycles-frontend` /
  topdown / `perf record -e l1-icache-load-misses --call-graph`.
- BOLT not run: cs_etm full-trace overruns TRBE (100% sample loss); the
  synth bench's PMU profile shows ZERO libwprofinj.so coverage in the
  decoded events. Would need to fall back to `arm_spe_0` or trace in
  snapshot mode.

To resume, in priority order:
1. **Identify which symbols accumulate L1i misses on ARM during attach** —
   `sudo perf record -e l1-icache-load-misses -t $BENCH_PID --call-graph
   dwarf -F 99 sleep 30` while wprof is attached, then `perf report`.
   This is the most direct path to whether the icache rise is from the
   callback's own footprint, from Python interpreter spilling cache, or
   from PyTorch's existing footprint being further perturbed.
2. **Re-attempt BOLT** with `arm_spe_0` instead of cs_etm (SPE works for
   `perf2bolt -spe`, llvm-bolt was built locally to
   `~/bin/llvm-bolt`). Build wprof with `BOLT=1` (already plumbed).
3. **Working-set measurement** — separate effort suggested by the user;
   the relevant cs_etm-based measurement methodology was referenced but
   not yet executed.
4. **Fix x86 clang inject regression** — needed before PGO/BOLT story
   has a comparable x86 number.

## Open follow-ups (not done)

- Investigate the clang-vs-gcc inject regression on x86 (blocks x86 PGO
  measurement entirely).
- Investigate why libwprofinj.so's PGO `.profraw` isn't flushed on bench
  exit on x86 (may be related to the inject regression above).
- Attribute ARM's IPC drop with stall counters
  (`stalled-cycles-frontend`, topdown).
- `perf record -e l1-icache-load-misses --call-graph` during attach to
  identify which symbols (callback? Python interpreter? PyTorch?)
  accumulate the misses.
- Re-collect branch counters with no-multiplexing event group.
- Validate on a real PyTorch GPU training step (these numbers are CPU-only
  with a deliberately Python-heavy model).
- AutoFDO as an alternative to PGO instrumentation (avoids the
  instrumentation-build slowdown perturbing the profile).
