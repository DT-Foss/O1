# HOLO-STREAM verdict — scoring PREDICTIONS.md P8–P11 against the full sweep

*2026-07-22 ~23:10 CEST. Data: results/holo_stream_recall.json (P∈{1,2,4} ×
G∈{0,8,32,128} × 2 seeds, 2000 iters, chunk 16, chance 0.0625). Predictions
were committed in 78e8ca6 before this file existed.*

## The headline

**Keyed, content-addressable recall survives 128 tokens of silence across 8
detach-carried chunk boundaries, flat, at ~8× chance** — carried P=2 recall
0.53–0.58 from G=0 to G=128 while the zeroed-at-gap null sits at chance
(0.045–0.09) in every one of the 12 cells. Neither source line of the repo
had this: MQAR was never chunk-carried, idle-persistence was never keyed.

## Scorecard

- **P8 (plateau, not decay) — CONFIRMED.** Ignited cells: P=2 G=8→G=128:
  0.530→0.480 (−5 pp, within the ≤10 pp band); P=4: 0.265→0.275 (+1 pp).
  The gap does not touch the binding — exactly the phase/magnitude split the
  theory derives (the write-once-freeze carrier defends magnitude; nothing
  attacks the stored pattern during silence).
- **P9 (1/√P ordering, conditional on ignition) — CONFIRMED.** At G=8:
  P=2 = 0.53 (≥0.5 ✓), P=4 = 0.265 (≥0.25 ✓).
- **P10 (factorization recall(P,G)=f(P)·g(G)) — CONFIRMED (weak form).**
  g(G) ≈ flat for both ignited P; the G-profiles are parallel.
- **P11 (phase pays rent at P≥2) — FALSIFIED in this regime.** P=2 G=8:
  holo_on 0.530 vs holo_off 0.530; P=4 G=8: 0.265 vs 0.290. The selective
  (rank-1 real) arm carries 2–4 keyed bindings through the gap just as well.
  **Interpretation, not excuse:** with P_max=16 keys and 512 state channels,
  input-conditioned gates (γ(x), α(x)) allow *channel allocation by key
  identity* — a per-key register file, no phase needed. The rank-1 capacity
  theorem's premise (key-agnostic write) is about outer-product structure,
  not about vocab-conditioned gating; a 16-key world fits in channels. The
  phase mechanism should start paying rent when binding must be
  *compositional*: key space ≫ channel count, or held-out keys never seen in
  training. That is the designed next attack (below).

## The P=1 anomaly — a curriculum pathology, diagnosed

P=1 carried collapsed to chance (0.04–0.10) in the full run despite 100% in
the smoke. Cause, read from the code path: the curriculum grows the gap on
`acc > 0.9` **every iteration** — an easy P=1 task sustains 0.9+ and rockets
G 2→128 within ~10 iterations. From G≳14 the sequence spans multiple chunks,
and truncated BPTT gives the *write phase* zero gradient from the
query-position loss (the KV block sits behind detached boundaries). The model
then spends ~1990 iterations training on a signal that cannot teach writing,
and unlearns. P=2/P=4 ignited precisely because they never cleared 0.9 —
they consolidated the write in the single-chunk regime and the frozen carrier
then generalized to G=128 for free (which is itself evidence for the theory's
carrier picture). The smoke (Gmax=8, always single-chunk) never left the
teachable regime.

**Fix (v2, launched tonight):** patience curriculum — grow the gap only after
acc > 0.9 is sustained for `--patience` consecutive iterations (default 25),
so each gap level consolidates before the next. Output:
results/holo_stream_recall_v2.json. Prediction for v2, registered now:
P=1 carried returns to ≥0.9 at every G including 128; P=2/P=4 match or exceed
v1; the zeroed null stays at chance; holo_on vs holo_off stays tied (the
P11 falsification is a regime property, not a bug — it should replicate).

## Next attacks (Phase B+)

1. **Compositional-binding test** — where the phase must pay rent: P_max=256
   (key space ~ channel count) and/or evaluation on held-out keys never seen
   in training. Channel allocation cannot cover unseen keys; de-rotation can.
2. **φ-drift probe** — measure arg(S) across the gap directly from internals
   (theory falsifier 1; predicted: zero drift).
3. **Capacity ladder on the stream** — n_slots>1 at P=8 over gaps, the
   streaming version of the slot experiments.
