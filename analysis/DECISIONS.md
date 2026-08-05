# POS — Design Decisions (Phase A build log)

One line per decision, in order. Context: BRIEFING_POS.md + David's session directives.

- **Threads = 1** (not repo's ncpu−2): measured under the live SAT-solver load, 8 threads is ~30× slower than 1 for these tiny ops (fwd+bwd 1436 ms vs 46 ms idle-sample; median ~1.2 s under load either way), and 1 thread is bit-deterministic → G2.
- **B=8, K=64 kept** (exact streaming_train recipe) despite B=64 benching ~5× more tok/s: chunk-mean surprise over 512 tokens is the committed gating signal; batching 32+ parallel document streams would average the signal away, and G1 parity to streaming_train (8.69→5.22 @3M) is literal with B=8.
- **Expected volume ≈ 30M streamed tokens in 40h** under current machine load (~220 tok/s ensemble measured); thesis needs ~3M+ (the whole published streaming_train curve) → 10× headroom; tok/s logged live in status.json, Phase B reads actuals.
- **Eval cadence = wall-clock 900 s** (200k-token frozen WT-2 val slice, ~56 s for 4 arms measured): throughput varies with external load, token-scheduled evals would swing 15 min↔hours; smoke/G2 use `--eval-every-tokens` (deterministic token scheduling).
- **G2 determinism gate = sha256 digest** over all chunk/eval metric fields (wall-clock/RSS excluded), printed at run end; two smoke runs must produce identical digests.
- **A3 gating mechanics**: forward always no_grad (briefing wording); gated chunks recompute the forward WITH graph from the same pre-chunk state (identical values, weights unchanged) then backward+step; threshold = q-quantile (q=0.80) of a 500-chunk rolling window of chunk-mean surprises, window updated AFTER the gate decision; first 100 chunks always backward (ignition) and counted in gradient tokens honestly.
- **Index warmup = 5M streamed tokens** (David's ~5M guideline; ≈6 h at current load, A3 has ~1.2M grad tokens by then and the surprise curve is past its steep fall) — final call confirmed after the smoke surprise curve.
- **Index/probe RNG isolated** in a dedicated torch.Generator (seed 42) so probes never touch arm determinism; probes run on cloned, row-sliced states (side-trips, never the live stream state).
- **Twin fork at wall-clock 24 h** (briefing): weights copied, Z=None + fresh Adam; twin INHERITS A3's rolling window and gates immediately (no ignition) — its elevated early gating is itself the measured warmup cost of a restart.
- **Reset-twin heldout == A3 at fork instant by construction** (stateless eval, same weights); the warmup cost shows in online stream surprise and post-fork gradient spend — both logged per chunk.
- **C4 stream resilience**: doc-counting wrapper with reconnect + ds.skip(docs) on network error (backoff ≤300 s); reconnects logged in status.json; exact data order guaranteed for smokes, best-effort across reconnects in the 40 h run.
- **Machine guard = pause, not kill** (briefing): RSS>12 GB or disk<5 GB → checkpoint, status "paused", re-check every 60 s; RSS-pause exits cleanly after 30 min (own RSS won't shrink by waiting); disk-pause waits indefinitely. NOTE: only ~6 GB free at launch — flagged to David.
- **Smoke outputs are tag-isolated** (`results/pos_<tag>_*`) so gates never clobber the real run's files.
- **q calibrated 0.80 → 0.75 after G3**: at q=0.80 the tail gate fraction measured 13.3% (the falling loss trend pushes chunks below the rolling window's quantile); recheck at q=0.75 measures 19.4% post-ignition / 15.3% tail — in the 15–30% band, and the fraction drifts toward nominal 25% as the curve flattens.
- **Probe logging 6 decimals + lookahead 32→16**: measured the injection effect on the young model at ~1e-4, concentrated in the ~6-token receptive field — 4dp logging would round it to zero and a 32-token lookahead dilutes it 2× (closed_loop precedent: 12); the state effect itself is real (max|ΔZ| = 1.29 between injected and random advance).
- **WP4 built and launched live** (David's go, revised from the earlier defer): src/holo_stream_recall.py marries the holographic complex write with detach-carried streaming state; equivalence full-vs-chunked < 1.2e-6; smoke P=1: carried recall 100% through gap 8 across chunk boundaries, zeroed-at-gap null at chance — the --full sweep (P∈{1,2,4}×G∈{0,8,32,128}, 2 seeds) runs nice-19 beside the long run; per David the historic ~9% MQAR ceiling is one agent-day of exploration, not a law — no ceiling assumption in the code.
- **Smoke checkpoints (47–67 MB each) deleted, not committed**: runtime artifacts only; every gate is evidenced by the committed jsonl/json logs; the long run's pos_ckpt.pt stays untracked (resume-only).
- **verify_pos.py two-tier**: hard PASS/FAIL only on data integrity + internal consistency; thesis numbers (A3/A2 ratio, injection deltas, twin warmup) are computed and printed as measured headlines — deviations are data points, not failures (briefing: Zielbild is orientation, not stop criterion).
- day4 ~10:30 — METHOD NOTE (rank-sweep builder's find): torch thread count
  changes TRAINING DYNAMICS near ignition boundaries, not just speed — the
  phase arm's 8.9% MQAR ceiling ignites under default multithreading (5/5
  historic seeds + 1/1 rebuild) but dies at threads=1 (0/4 seeds, ~4 sigma
  below reference). Reduction-order sensitivity in the early unstable phase.
  Consequences: (1) instruments measuring against a historic reference must
  reproduce the reference's threading regime; (2) all existing knee-arc /
  POS results remain internally valid (every arm-vs-arm comparison ran in
  ONE regime — fairness holds), but cross-regime numeric comparisons are
  not meaningful; (3) rank_sweep.py runs at torch default threads; beast
  striping compensates with fewer parallel stripes (4 stripes x 4 threads).

## Standing measurement rule: the cadence axis (2026-08-05, MS-G audit)

Any metric with per-chunk cost in its denominator — stall ratios, replay
speed, snapshot cost in "chunk slots", tok/s comparisons — is meaningless
unless (batch, chunk, d_model) are (a) set explicitly by the run and
(b) recorded IN THE SAME ARTIFACT the metric ships in. Width alone does not
make a chunk compute-heavy: chunk WEIGHT is B×K, and a subprocess that
inherits module defaults silently measures the defaults, not the system.
This killed P39 (a)/(c) twice — first as toy-cadence inflation (17.9×/3.79×
→ 7.1×/2.2× at real cadence), then as a real residual that falsified both
checks. Full audit: results/cadence_audit.json. Enforcement: portable_organism
and moebius_stage now expose and forward --batch/--chunk-size; every new
harness that reports a cost ratio must embed its cadence config in its
result JSON.
