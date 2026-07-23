# NEXT — Phase B (T+48h): harvest the POS long run

## The one command

```bash
claude --model claude-opus-4-8 --dangerously-skip-permissions \
  -p "Führe Phase B aus NEXT.md aus: pos_analyze.py laufen lassen, Plots
  erzeugen, verify_pos.py grün machen, analysis/POS_THESIS.md (≤1 Seite,
  Zahlen aus results/) schreiben, committen."
```

## What is running (started Phase A, 2026-07-22 evening CEST)

One detached CPU process, `src/pos_run.py --hours 40 --q 0.75` — three arms
(A1 forward-only, A2 full-gradient, A3 surprise-gated, all StreamingNoPELM,
identical init seed 42) on one cloned C4 stream, plus the A3 index loop
(spike→span store from 5M tokens, paired injection probes on ≥2h key
recurrence) and the reset-twin A3R forked from A3 at wall-clock 24h.

- Heartbeat: `results/pos_status.json` (pid, phase, n_streamed, tok/s, index stats).
- Raw logs: `results/pos_metrics.jsonl`, `pos_chunks.jsonl`, `pos_index.jsonl`,
  `pos_probes.jsonl`; run log `results/pos_run.log`; checkpoint `results/pos_ckpt.pt`.
- **--hours 40, not 48** (briefing allows the call): the machine is saturated by a
  SAT-solver fleet (~220 tok/s ensemble → ~30M tokens, 10× the published
  streaming_train curve); 40h gives ≥16h of twin comparison after the 24h fork,
  ~34h of index operation after the ~6h warmup, and finishes with margin before
  T+48h so Phase B reads a *completed* run. Build gates G1 (parity), G2
  (bit-identical digests), G3 (gate fraction, q calibrated 0.80→0.75) all passed
  before launch — see analysis/DECISIONS.md for every design call.

## Phase B steps

1. Confirm the run finished: `results/pos_status.json` phase == "done"
   (if "running", let it finish; if it died, `python src/pos_run.py --resume
   --hours <remaining> --q 0.75` continues from the checkpoint, logs auto-trimmed).
2. `python src/pos_analyze.py` → 6 plots in plots/ + `results/pos_summary.json`.
3. `python src/verify_pos.py` → must exit 0 (INTEGRITY hard, HEADLINES informational).
4. Write `analysis/POS_THESIS.md`, ≤1 page, every number from results/:
   - the core claim: A3's fraction of A2's heldout improvement at its fraction of
     gradient tokens (`ratio_A3_vs_A2_improvement`, `A3_grad_token_frac`);
   - flat RSS (`rss.span`), one process, zero restarts (`stream_reconnects` is
     network-only);
   - injection: `mean_d_inj` vs `mean_d_rand` + helped fractions (paired, 6-decimal
     logs; effect is small on a young model — the interesting axis is whether it
     grew as the γ-spectrum matured);
   - twin: `surprise_excess_first_2h`, post-fork gate fractions (A3R over-gates =
     the warmup a restart pays), `converged_at_n`.
   Deviations from the Zielbild are measurements, not failures — characterize them.
5. Commit (local, no push).

## Phase B+ — holographic recall on the living state (WP4, already ignited)

`src/holo_stream_recall.py` marries the holographic key-conditioned write with
detach-carried streaming state (first time in this repo). Smoke: equivalence
full-vs-chunked < 1e-5, P=1 recall **100% through gap 8 across chunk boundaries**,
zeroed-at-gap null collapses to chance (~6%). The `--full` sweep
(P∈{1,2,4} × G∈{0,8,32,128}, 2 seeds) was launched nice-19 alongside the long run
→ `results/holo_stream_recall.json` (+ log `results/holo_stream_recall.log`).
In Phase B: read the JSON, report the (P,G) ignition table and whether holo_on
separates from holo_off at P≥2 (the smoke was iteration-budget-limited there —
per David, do NOT treat the historic ~9% MQAR ceiling as a law; the streaming-gap
regime is new territory). If the full sweep is still mid-flight, note it and
harvest what's there.

## Run-time observations Phase B must know

- **The index hit its 20,000-entry cap by n≈11M** (~30 min after the 5M
  warmup): spike_min_nll=7.0 is permissive once loss ≈ 5.3, so ~2
  spikes/chunk filled it immediately. The stored spans are therefore an
  early-run snapshot (≈5–11M-token era); recurrence probes test *those* keys
  for the rest of the run. Interpret injection results with that provenance
  in mind; a v2 of the index would use a rolling/adaptive spike threshold.
- Machine load dropped overnight → tok/s roughly doubled (~300→~660); all
  token-axis scheduling (evals per 15 min wall-clock) is unaffected.

## Known environment facts (do not re-derive)

- torch threads MUST stay 1 (measured: 8 threads is ~30× slower under the
  SAT-solver load; also bit-determinism).
- Wall-clock numbers (tok/s) are contaminated by external machine load; every
  scientific axis is token-based. Never claim throughput.
- Disk hovers near the 5 GB pause threshold; the runner pauses+resumes cleanly
  (it did so twice in smokes). Pauses appear in status.json, cost nothing.
