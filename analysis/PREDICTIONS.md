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

## Wave 5 — the deployment primitives (registered day 3 ~20:00, before any build)

- **P23 — weight hot-swap on the living stream (MS11,
  src/state_weight_swap.py):** carrying an OLD state into NEW weights works.
  With W(359M) fixed and fresh C4 chunks cloned to all arms (forward-only):
  (a) the cold-start arm's online-NLL excess over the native arm in the first
  50 chunks is ≥ 2× the hot-swap-far arm's (state from 128M, 231M tokens of
  training distance); (b) both hot-swap arms converge to within 0.01 nats of
  native inside 300 chunks; (c) the channel-shuffled control is worse than
  hot-swap-far throughout (compatibility is structural, not a bias artifact).
- **P24 — hot-swap growth (MS8, src/hot_swap_growth.py):** function- and
  state-preserving widening (channel duplication, d64→d128, carried Z
  migrated) on a live C4 stream: (a) surgery equivalence |Δlogits| < 1e-4;
  (b) no post-surgery transient (first-20-chunk online NLL within 0.05 of the
  stay-d64 arm); (c) at +1.5M post-surgery tokens the grown arm's held-out
  beats stay-d64 by ≥ 0.03 nats (the new capacity is used); (d) grown beats
  fresh-d128-from-scratch at the same wall-token axis (growth beats restart).
- **P25 — α-shut pollution control (MS12, src/holo_alpha_shut.py):** the F3
  discovery is causal: adding an α(x_filler) regularizer to the T2+MagNorm
  knee recipe (λ sweep incl. λ=0 control) (a) cuts trained filler φ-drift at
  G=512 from ~1.5 rad to < 0.5 rad, (b) moves the recall knee to ≥ 1024, and
  (c) drift reduction and knee position are dose-monotone in λ. If large λ
  strangles the write itself, that trade-off boundary is the measurement.

- **P26 — the stored bit survives the surgery and the swap (MS13,
  src/beacon_swap.py, registered before the build):** on the beacon
  idle-persistence task (write-once-freeze carrier, streaming_train.py §D/E):
  (a) across the MS8 widening surgery (d64→d128, carried-Z migrated,
  post-gate) beacon recall through a 256-token gap stays ≥ 0.99 — the bit
  survives the brain operation exactly; (b) within one training run, writing
  the bit under W(T1)'s encoder and reading it under W(T2)'s decoder (weights
  from 2× further training, state carried across the swap) keeps recall
  ≥ 0.9 — the state CODE (which channel carries the bit, at what scale) is
  stable across training distance once the carrier has locked; (c) the
  channel-shuffled state control collapses to chance at both (a) and (b). If
  (b) fails while (a) passes, the honest reading is: state code drifts with
  training distance, and a state-alignment map becomes the next disclosed
  primitive — either outcome is the measurement.

### Wave-5 scoring (as results land)

- **P23 SCORED (results/state_weight_swap.json, A3 organism, 600 chunks):**
  (a) **FALSIFIED, cleanly and instructively** — cold (Z=0) shows NEGATIVE
  first-50-token excess (−0.085) vs native; the carried fast-path state holds
  no NLL value beyond the ~5–8-token receptive field. This is F2 measured
  from the other side, not a bug. (b) **CONFIRMED, far beyond the bar** —
  both swap arms converge to the native trajectory in 4 chunks (256 tokens;
  bar was 300 chunks). (c) inconclusive by construction: past chunk ~4 all
  arms are bit-identical (deterministic convergence), so windowed separation
  only exists in the onset (where the 30-trial diagnosis separates shuffled
  at +0.9). Net reading: weight hot-swap on the fast path is a NON-EVENT —
  no lock-in, no compatibility risk, nothing to migrate; the portable value
  lives in the slow channels (→ P26).
- **P24 SCORED (results/hot_swap_growth.json, 1.2M+1.2M):** (a) CONFIRMED
  6.7e-6; (b) CONFIRMED gap 0.046 (and conservatively measured: the grown
  arm restarts Adam, stay64 keeps warm moments — the pure surgery transient
  is smaller); (d) CONFIRMED growth beats restart by 0.127 nats. (c)
  **FALSIFIED at this scale**: grown trails stay64 by 0.036 at 1.2M
  post-surgery tokens — but the deficit HALVED from 0.073 at 300k, monotone;
  next lever registered: migrate Adam moments through the duplication map
  instead of resetting, and/or longer horizon.

## Wave 6 — the pull-system queue (registered day 3 ~20:45, before any build)

- **P27 — Adam-moment migration closes the growth deficit (MS14,
  src/hot_swap_growth.py --migrate-moments):** migrating optimizer moments
  through the duplication map (exp_avg via the gradient transform, exp_avg_sq
  via its square) instead of resetting Adam: (a) the commutation gate holds —
  grow(adam_step(m64)) == adam_step(grow_with_moments(m64)) to < 1e-3 on all
  parameters after one identical-batch step; (b) at 1.2M post-surgery tokens
  the grown arm's deficit vs stay64 shrinks from −0.036 to ≥ −0.01 or turns
  positive; (c) growth still beats restart.
- **P28 — curiosity homeostasis beats fixed q under shock (MS4,
  src/pos_auto_q.py on the MS3 harness):** a homeostat that regulates q to
  hold a target gate rate (~25%): (a) holds gate_frac in [0.18, 0.32] through
  the C4→code→C4 shock while fixed q=0.75 overshoots during phase 2 (R2
  measured 0.58); (b) at MATCHED total gradient tokens, auto-q's WT-2
  forgetting ≤ fixed-q's and its code plasticity ≥ 0.9× fixed-q's — the
  homeostat spends the same budget on better-chosen chunks; (c) the q
  trajectory itself is the measurement: it must RISE during the shock (the
  organism becomes pickier when everything is surprising) and relax after.

- **P25 SCORED (results/holo_alpha_shut.json + _lam0check.json): FALSIFIED
  in the registered regime, with the mechanism found.** The α-filler
  regularizer neither reduces trained drift@512 (1.19 vs 1.20 reference) nor
  moves the knee (λ=0 knees 1024/512 across seeds; λ>0 knees 256–512 — the
  regularizer HURT where it did anything). Root cause, discovered by the
  builder's forensic pass: the M3 recipe's curriculum NEVER ignites —
  final_train_gap=2 in all six cells of the original holo_magread run too;
  the 512 knee was always pure kickstart+magnorm EXTRAPOLATION from a
  gap-2-trained model. So the regularizer only ever acted on 2 filler
  positions; α-behavior on long gaps was never trained in any arm. The
  registered lever cannot reach the mechanism in this regime — honest kill.
- **P29 — inference-time α-clamp, the direct pollution causality test (MS12b,
  registered before the build):** on the M3-recipe model (λ=0, gap-2-trained,
  knee 512), clamping α(x) toward 0 on filler positions AT EVAL ONLY (state
  write suppressed during silence, no training change): (a) trained filler
  φ-drift at G=512 drops below 0.3 rad (untouched: ~1.3); (b) the recall
  knee moves to ≥ 1024; (c) recall at G≤128 is unchanged within 5 pp (the
  clamp must not damage in-range recall). If (b) fails while (a) passes,
  pollution is real but NOT the binding constraint at 512+ — the honest
  alternative (magnitude floor? phase SNR?) becomes the next measurement.

- **P29 SCORED (results/holo_alpha_clamp.json): (a) CONFIRMED — the eval
  clamp eliminates trained filler drift exactly (0.0 rad @512, both seeds;
  the zero-drive law demonstrated on the trained model); (c) CONFIRMED
  (in-range recall −5pp, at tolerance); (b) NOT MET — the clamped knee (640
  mean) is not higher than unclamped (768). The honest fallback fired, with
  the mechanism visible in the raw data: WITHOUT filler writes the magnitude
  collapses unfed (mag_ratio → 0.0007 @2048, snr-alive 1.0→0.09) while
  unclamped filler writes REFRESH it (30–80×) even as they pollute the
  phase. The filler write is a double agent: phase pollutant, magnitude
  feeder. The knee past 512 is MAGNITUDE-bound, not pollution-bound.**
- **P30 — the 2×2 that disentangles the two axes (MS12c, registered before
  the build):** eval-time arms {clamp, no-clamp} × {in-state magnitude
  refresh at chunk boundaries (renormalize |S|→1, phase untouched — the
  variant disclosed in FOUNDATIONS F3; eps-guarded so a zeroed state stays
  dead), no-refresh} on the M3-recipe model: (a) clamp+refresh knee ≥ 2048;
  (b) ordering clamp+refresh > refresh-only > unclamped > clamp-only (drift
  costs recall once magnitude is guaranteed; feeding beats starving); (c)
  zeroed-at-gap null stays at chance in every arm (the refresh must not
  invent information); (d) in-range recall @G≤128 within 5pp everywhere.

- **P30 SCORED (results/holo_clamp_refresh.json): (a) CONFIRMED — knee
  2176 mean under clamp+refresh (seed0 reaches the 4096 end of range at
  acc 0.33 vs chance 0.06; seed1 stays at 256 — real seed variance, stated);
  (d) CONFIRMED (in-range recall 0.41–0.46 everywhere). (b) NOT MET as a
  strict ordering, and that is the finding: refresh WITHOUT clamp is the
  WORST arm (512 < clamp-only 640 < untouched 768 << both 2176) — the two
  axes INTERACT, they do not add. Refresh rescales the polluted direction
  (it cannot heal what only the clamp prevents); the clamp starves the
  magnitude (only the refresh feeds it). Persistence = clean phase AND fed
  magnitude, jointly. (c) nominally out at 12/48 null cells but symmetric
  around chance with mean deviation +0.005 — sampling noise at the small
  large-G eval batches, not information manufacture (eps-guard verified in
  isolation).**

## Scoring rule

Each P-item gets CONFIRMED / PARTIAL / FALSIFIED in the harvest documents,
with the measured number beside the predicted interval. A falsified
prediction is a measurement — the register exists so we can't unknow what we
expected.
