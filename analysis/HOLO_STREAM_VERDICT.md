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

## v2 (patience curriculum) — the registered prediction is FALSIFIED

*Added ~00:10 after results/holo_stream_recall_v2.json landed.*

v2 predicted P=1 carried ≥ 0.9 at every G with the patience curriculum.
Measured: P=1 collapses again (carried 0.045–0.085 ≈ chance; final_train_acc
0.06–0.09) — **even though** the curriculum this time consolidated before
growing (Gcur reached 26/40 via sustained >0.9 phases, so mastery at
single-chunk gaps demonstrably happened first). The sharpened diagnosis: the
failure is not growth *speed* but **structural gradient blindness** — past
the chunk boundary the query loss cannot reach the write phase at all
(truncated BPTT), and continuing to train there is ~1900 iterations of
full-momentum Adam on a signal that cannot teach writing: it catastrophically
forgets the consolidated behavior. Patience only delays the collapse.
(P=4 cells are bit-identical v1↔v2 — patience never triggered there — an
incidental determinism check that both runs share one trajectory until the
first curriculum decision diverges.)

**v3, launched tonight: train-short-eval-long.** Cap the *training* gap at
the single-chunk maximum (chunk − 2P − 2); large gaps become pure eval
extrapolation. This is the repo's core recipe applied to the gap axis (train
T=32 → eval 131k). v1's ignited cells already demonstrate it implicitly:
P=2/P=4 trained at G=2 and held flat to G=128. **v3 prediction, registered
before the run:** P=1 carried ≥ 0.9 at every eval G including 128; P=2/P=4 at
or above their v1 levels; zeroed null at chance throughout; holo_on vs
holo_off still tied (the P11 falsification is a regime property and should
replicate). Output: results/holo_stream_recall_v3.json.

## v3 (train-short-eval-long) — the fix works, and the phase starts paying rent

*Added ~01:05, scored against the v3 prediction registered above.*

- **P=1 repaired at G≤32:** curriculum consolidates at the single-chunk cap
  (final_train_acc 1.0), carried recall 1.000/1.000/0.995 at G=0/8/32.
  Prediction "≥0.9 at every G" is **PARTIAL**: G=128 breaks (0.065).
- **P=2 jumps to 0.79–0.83 at G≤32** (v1: 0.53) — best of all versions — and
  **holo_on now beats holo_off by +25–30 pp** (0.80 vs 0.50). The original
  P11 falsification was an artifact of collapsed training: with clean
  consolidation the key-conditioned phase write *does* pay rent at P=2, in
  streaming, across chunk boundaries. P11 status revised: falsified in the
  v1 regime, **confirmed in the v3 regime for G≤32**.
- **Zeroed null at chance in all 12 cells** — the carried-state claim stands
  in every version.
- **The new measurement: a γ-knee between G=32 and G=128.** v3's phase arms
  hold 0.995 at G=32 and die at G=128, while v1's G=2-trained P=4 arm stays
  flat to 128 and v3's holo_off P=1 extrapolates perfectly to 128. This is
  the theory's §1 knee (G* ≈ ln-margin/(1−γ)) made visible: the *training gap
  regime shapes how tightly γ closes on fillers* — training at larger
  single-chunk gaps (≤12) appears to license faster forgetting than training
  at G=2. The knee's location (32 < G* < 128) and its dependence on the
  training gap are now concrete, measurable quantities.

**Where this leaves the mission:** keyed holographic recall on the carried
streaming state is real (0.8 at P=2 through 32-token gaps, +27 pp over the
magnitude baseline, null at chance), and the open frontier is a measured
decay knee, not a mystery. Phase B+ items below, updated accordingly.

## Day-1 moonshot M1: held-out keys — P12 falsified, and the finding underneath

*Added ~07:45. Data: results/holo_heldout_keys.json (2000 iters) and
_8k.json (8000 iters, the consolidation control); P12 was registered in
ef93620 before the runs.*

- **Both arms generalize to never-trained-as-key ids** — generalization gap
  ≈ 0 (−0.04…+0.08) against a 0.49 margin over chance, at 2k AND 8k iters,
  null at chance. The P12 dichotomy (phase = function vs baseline = lookup
  table) was wrongly posed: the magnitude arm's binding is ALSO a function of
  embedding geometry, not a per-key register file. Channel-allocation-as-
  lookup is dead as an explanation of anything in this series.
- **The v3 phase advantage does not appear at P_max=64** (both arms ~0.55 vs
  v3's 0.80/0.53 at P_max=16), and this is NOT a budget artifact: 4× the
  iterations produced no consolidation trend (final_train_acc 0.38–0.63,
  curriculum never left G=2). In this config (d=64, V=16, 64 keys) the task
  plateaus for both mechanisms. Open question for Phase B+: is the phase
  advantage a small-key-space phenomenon, or does it return with capacity
  (d_model, n_slots) scaled to the key space?

## Day-1 moonshot M2: the γ-knee — P13 point targets missed, theory law confirmed, blocker relocated

*Added ~08:15. Data: results/holo_knee.json (bar 0.9) and holo_knee_bar08.json
(bar 0.8); P13 registered in ef93620.*

- **P13 point targets NOT MET** (T2 knee 192 vs ≥256 predicted; T3 192 vs
  ≥1024) — but the knee moved from v3's 32<G*<128 to a solid **256 in the
  best cells (0.37 acc at G=256, null at chance)**, and the **theory's
  ordering law held in both runs**: knee order tracks filler-γ order exactly
  (γ 0.87→knee 80; γ 0.988→knee 192).
- **Three blockers found and peeled in sequence:** (1) the 0.9 curriculum bar
  never fires for P=2 (consolidates ~0.85) → T1≡T2 bit-identical, the
  full-seq mechanism untested; bar=0.8 activated it (seed 1 grew G_train to
  26 under full-sequence gradient — first time in this series). (2) Ignition
  is seed-variant (seed 0 never clears 0.8). (3) With the per-CHANNEL γ
  metric (the Pillar-E "the average hides the carrier" lesson, now logged):
  **carrier channels at τ≈160–700 already exist in every variant** — γ is no
  longer the bottleneck. Knee 256 despite a τ=700 channel implies the rms
  READOUT tolerates only ~30% magnitude loss (G* = τ·ln(S₀/S_min) with a thin
  margin). **The next lever is readout margin, not γ** — e.g. a
  magnitude-normalized read (the de-rotation phase is intact; re-normalizing
  |S| before the read should be nearly free) — Phase B+ item.

## Day-1 second wave (P14–P17) — scored

*Added ~09:00. All predictions registered in 38a154c before the runs.*

- **P14 (mag-normalized read, M3): point target missed, arc extended.** Knee
  at **512 in both seeds** (V2 kickstart+magnorm) vs 128/256 reference; the
  read fix is synergistic with large γ (V3 magnorm-alone stays at reference —
  exactly what theory demands) and costs nothing near-field (P14c ✓). The
  one-day knee arc: 32–128 (v3) → 256 (M2) → **512** (M3), every doubling
  theory-led; best channel τ≈1001.
- **P16 (state+index hybrid, M5): PARTIAL, mechanism fully validated.**
  Hybrid lifts P=16 recall 0.15→0.51 and P=8 0.20→0.66, dose-dependent
  (gate 0.5 intermediate), random-injection guard holds (wrong value
  actively hurts). Ceiling diagnosed: an untrained-consultation read tops
  out ~0.84 even with perfect injection — **the model never learned to be
  reminded**. Next lever: consultation-in-training.
- **P17 (capacity return, M6): WEAK PARTIAL after the lr-control.** The
  d=256 holo_off collapse was an lr artifact (best-of-off at lr=1e-3:
  0.50–0.57). Against best-of-off, the phase buys **+13–15pp on trained
  keys at G≤8 only**; on held-out keys +2/−6pp, at G=32 negative. Combined
  with M1: the phase advantage is now mapped as a narrow-regime phenomenon
  (16 keys: +27pp; 64 keys d=64: 0; 64 keys d=256: train-only +13pp).
  The control discipline shrank a +41pp smoke mirage to its honest size.
- **P15 (wake/sleep cycles, M4): v1 FAIL diagnosed** (replay overfitting:
  8–10 spans/cycle × 10-chunk sleeps ≈ 20 epochs over ~24 spans — the
  opposite regime of pos_sleep's 20k-span <1-epoch win); v2 with
  volume-coupled sleep budget (epoch cap, fairness-preserving reallocation)
  in flight — scored when it lands.

## F3 φ-drift probe — the law locked, and the knee reinterpreted

*Day 2 ~19:30, results/phi_drift.json.* Null-input control: with the drive
forced to exactly zero, |Δφ| ≈ 1e-8 across G≤512 in trained AND untrained
models — HOLO_CARRIER_THEORY §1 holds at machine precision (an atan2-on-
dead-channels measurement artifact was caught and SNR-gated; raw numbers
kept in the JSON). The deployment measurement is the discovery: real fillers
are NOT silence — α(x_filler) writes actively (magnitude grows up to 35×,
not decays), and the real phase drift (0.1 rad at G=1, 1.52 rad at G=512
trained) is ACTIVE POLLUTION, not decay. This reinterprets the knee: the
G=128→512 falloff is at least partly filler pollution — which is why the
magnitude-normalized read (heals decay, not pollution) stopped at 512. Next
theory-led lever: α-shut on fillers (the measured Pillar-E write-once-freeze
recipe) as an explicit curriculum/regularization target.

## Next attacks (Phase B+)

1. **Compositional-binding test** — where the phase must pay rent: P_max=256
   (key space ~ channel count) and/or evaluation on held-out keys never seen
   in training. Channel allocation cannot cover unseen keys; de-rotation can.
2. **φ-drift probe** — measure arg(S) across the gap directly from internals
   (theory falsifier 1; predicted: zero drift).
3. **Capacity ladder on the stream** — n_slots>1 at P=8 over gaps, the
   streaming version of the slot experiments.


## The knee arc, closed (P25/P29/P30 — day 3 evening)

The alpha-regularizer is an honest kill (P25: the M3 recipe trains at gap 2
and extrapolates — discovered in the forensics — so the lever never reaches
the mechanism). The eval-time causal chain then closed the story: the filler
write is a DOUBLE AGENT — it pollutes the phase (removable exactly: drift
0.0 rad under an eval clamp, P29a) and feeds the magnitude (clamped, the
channels starve: mag_ratio 0.0007, snr-alive 0.09). Neither lever alone
moves the knee (clamp-only 640, refresh-only 512, untouched 768) — together
they move it to **2176 mean / 4096 end-of-range** (P30a; seed variance
stated). Persistence = clean phase AND fed magnitude, jointly — an
interaction, not a sum. Files: results/holo_alpha_shut.json,
holo_alpha_clamp.json, holo_clamp_refresh.json.