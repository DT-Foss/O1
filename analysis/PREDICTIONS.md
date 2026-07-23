# PREDICTIONS — committed before the data lands

Registered 2026-07-22 ~22:15 CEST, while the POS long run (started ~21:45, PID
39826) and the WP4 holographic-gap full sweep are mid-flight. Everything below
is falsifiable by files that do not exist yet. Knowledge cutoff for these
predictions: the build-gate smokes (results/pos_g3b_*, pos_mech_*,
holo_stream_recall_smoke*.json) — nothing from the runs in flight.

The point of this file: tomorrow's numbers either confirm or kill these — no
story-fitting after the fact. Verdicts go into POS_THESIS.md / the WP4 note
with explicit references back to each P-number.

## POS long run (results/pos_summary.json, T+40h)

- **P1 — the core ratio.** A3 captures **0.80–0.95** of A2's heldout
  improvement (point estimate **0.85**) at a gradient-token fraction of
  0.18–0.26. Basis: surprise-selected chunks carry above-average learning
  signal per token; smoke ratios ~1.0 were ignition-dominated and don't count.
  Strong-form falsifier: ratio < 0.75. Embarrassment threshold the other way:
  ratio > 1.0 sustained (gating *beats* full gradient per streamed token)
  would be a bigger result than the thesis itself.
- **P2 — gate drift.** Post-ignition gate fraction starts ~0.15 and drifts
  toward the nominal 0.25 (=1−q) as the loss curve flattens; cumulative final
  in **0.18–0.26**.
- **P3 — flat RSS.** Post-warmup RSS span < **0.15 GB** over the whole run;
  zero process restarts (stream_reconnects counts network only).
- **P4 — frozen control.** A1 heldout constant at 8.6656 ± 0.001 for 40h.
- **P5 — the twin signature.** At fork: A3R heldout == A3 (same weights, by
  construction). Then three transients: (i) online surprise excess s3r−s3 in
  the first 2h between **+0.03 and +0.15**; (ii) A3R over-gates ≥ **1.5×**
  A3's post-fork gate rate in those 2h (the restart *pays extra gradient
  tokens* to rebuild what the living state carried); (iii) rolling |s3r−s3|
  converges below 0.05 within 2–8h. End-of-run heldout gap |A3R−A3| < 0.05
  (the warmup is a transient tax, not a permanent scar — that's exactly what
  makes it a *cost of restarting*, not a capability gap).
- **P6 — injection, paired.** mean_d_inj > mean_d_rand with the majority of
  probes d_inj > d_rand (sign test). Magnitudes small: mean_d_inj ∈
  [+0.001, +0.05], mean_d_rand ∈ [−0.005, +0.005]. Second-order prediction:
  the paired difference in the **second half** of the run exceeds the first
  half — the effect grows as the γ-spectrum matures (measured injection
  transport at 60k tokens was ~1e-4; closed_loop's trained-model figure was
  +0.026).
- **P7 — probe volume.** 100–600 probes total (30/h cap, ~5h warmup + 2h
  recurrence latency, C4 4-gram recurrence rates).

## WP4 full sweep (results/holo_stream_recall.json, tonight)

- **P8 — persistence axis (G).** For every (P, arm=carried) cell whose
  curriculum ignited: accuracy at G=128 within **10 pp** of accuracy at G=8 —
  a plateau, not an exponential decay. This is the theory's sharpest claim
  (the phase does not rotate during a gap; see HOLO_CARRIER_THEORY.md). The
  zeroed-at-gap null sits at chance (~0.0625) in every cell.
- **P9 — capacity axis (P).** Carried recall at G=8 orders as ~1/√P
  interference: P=1 ≈ 1.0, P=2 ≥ 0.5, P=4 ≥ 0.25 — *conditional on ignition*;
  ignition itself is the biggest uncertainty (2000 iters may not clear the
  0.9-curriculum bar at P=4; a stuck curriculum is a budget statement, not a
  capacity statement, and must be labeled as such).
- **P10 — factorization.** recall(P,G) ≈ f(P)·g(G): the G-shape is the same
  across P (correlation of normalized G-profiles > 0.9 across ignited P
  cells). If this holds, gap-persistence and pair-capacity are *independent
  axes* — the disruptive reading in the theory note.
- **P11 — the phase pays rent at P≥2.** holo_on − holo_off ≥ **+10 pp** at
  P=2, G=8 (rank-2 vs rank-1 per channel). At P=1 the two arms tie (both
  100%): one binding needs no key-conditioning — that tie is *predicted*, not
  a failure of the mechanism.

## Day-1 moonshots (registered ~07:20, before the --full runs)

- **P12 — held-out-key binding** (src/holo_heldout_keys.py, full run in
  flight): original form — holo generalizes to unseen keys (≥3× chance,
  ≥ off + 10pp), off drops to ~chance. The smoke already fired the
  pre-written alternative branch (BOTH arms generalize at 600 iters); the
  full-budget question is whether the v3 phase advantage (+25 pp) reappears
  on train keys and extends to test keys. Honest state: the channel-
  allocation lookup story is already wounded; embedding geometry may carry
  key identity for both mechanisms.
- **P13 — γ-knee mobility** (src/holo_gap_knee.py): full-sequence training
  (the equivalence theorem licenses train-unchunked/deploy-chunked) moves
  the recall knee from 32<G*<128 (v3) to ≥256; with a γ-kickstart head to
  ≥1024; knee position correlates with the measured filler-γ across
  variants (the theory's G* ≈ ln-margin/(1−γ) tested directly).

## Day-1 second wave (registered ~08:30, before any of the four runs)

- **P14 — magnitude-normalized read** (M3): the M2 blocker is readout margin,
  not γ (τ≈700 channels exist). Renormalizing |S| before the de-rotation read
  (phase is intact by theory §1) moves the knee from 256 to **≥1024**; zeroed
  null stays at chance.
- **P15 — closed wake/sleep cycles** (M4): iterated collect→consolidate
  cycles beat plain continued training at equal TOTAL gradient budget
  (final heldout lower by ≥0.02), and the per-cycle sleep dividend does not
  collapse to zero after cycle 1 (it is a mechanism, not a one-off).
- **P16 — state+index hybrid on MQAR** (M5): with a surprise-gated external
  index and paired injection at query time, recall at P=16 (far above the
  state's capacity) reaches **≥0.9** (state alone ≤0.15, chance 0.0625);
  random-injection control ≈ state alone; at P=2 the hybrid is not worse
  than state alone (the index must not hurt within-capacity recall).
- **P17 — does the phase advantage return with capacity?** (M6): at P_max=64
  (where M1 found holo==off at d=64), scaling d_model to 256 restores
  holo−off ≥ **+15 pp** at G≤32. If it does NOT return, the v3 phase
  advantage is a small-key-space phenomenon — scored honestly either way.

## Wave 3 — the dynamic portfolio opens (registered day 1 ~15:05, see analysis/MOONSHOTS.md)

- **P18 — learned to be reminded (MS1):** training WITH stochastic
  consultation (p=0.5 correct, p=0.1 wrong injection) lifts hybrid@P=16 from
  M5's 0.51 to **≥0.85**; base (no injection) stays within 5pp of M5's base
  (the skill is arbitration, not degradation); and on wrong injections the
  trained model loses LESS than M5's random arm did (it learns to weigh
  state against reminder).
- **P19 — the dream generator (MS2):** training on the model's own sampled
  continuations ("dreams", same gradient budget, warm opt) beats fresh data
  (self-distillation stabilizes: dream delta > fresh delta by ≥0.03) but
  loses to stored-span replay (real surprises carry information dreams
  cannot invent). If dream ≥ sleep instead: storage-free consolidation —
  MS6/MS7 redesign per MOONSHOTS.md rule 2.
- **P20 — domain shock (MS3):** on C4→code→C4, full-gradient learns code
  fastest but forgets most (WT-2 heldout degrades ≥0.15 during the code
  phase); surprise-gating alone forgets less at comparable code plasticity;
  gating + dosed replay of phase-1 spans forgets the least (≤50% of
  full-gradient's forgetting) — sleep as the anti-forgetting organ.

## Foundations track / wave 4 (P21–P22)

*P21 and P22 were written into their builder scripts' docstrings before the runs
launched; unlike P1–P20 they were committed together with the first harvest, not
in advance — recorded here with that caveat, scored with the same rigor.*

- **P21 — language-stream holographic graft (MS5,
  src/holo_language_graft.py):** the holo graft on the frozen 400M-token POS
  snapshot clears ≥3× chance (≥0.19, chance 0.0625) at G=128 on real text;
  the ctrl graft stays below that; the zeroed null decays to chance.
- **P22 — family transfer (F5, src/pos_family_transfer.py):** S6-POS-ratio ≥
  0.85 × GSSM-POS-ratio (family-generic gating); GSSM-full ≤ S6-full + 0.1
  nats (architecture-competitive, the DD baseline). **SCORED on the 6M full
  run (results/pos_family.json): CONFIRMED, both parts** — ratio-of-ratios
  0.9804 (GSSM 0.9523 at gate 22.6%, S6 0.9337 at gate 23.1%); head-to-head
  GSSM-full 5.177 vs S6-full 5.333 nats — GSSM *leads* by 0.156 nats at scan
  parameter parity 1.0016 and identical pipeline/tokens/seed.

## Scoring rule

Each P-item gets CONFIRMED / PARTIAL / FALSIFIED in the harvest documents,
with the measured number beside the predicted interval. A falsified
prediction is a measurement — the register exists so we can't unknow what we
expected.
