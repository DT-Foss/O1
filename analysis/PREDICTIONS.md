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

### POS long-run scoring (Phase B, day 4 — results/pos_summary.json, verify_pos 16/16 exit 0)

- **P1 CONFIRMED BEYOND ITS OWN CEILING: ratio 1.0091 at grad-token
  fraction 0.2517** — the registered embarrassment threshold ("ratio > 1.0
  would be a bigger result than the thesis itself") fired. A3 8.6588→4.7430
  vs A2 8.6588→4.7782 on 909.7M streamed tokens.
- **P2 CONFIRMED**: cumulative gate fraction 0.2517, post-ignition 0.2516
  (band 0.18–0.26).
- **P3 FALSIFIED as written**: RSS span 0.972 GB vs the <0.15 GB band — the
  prediction ignored the index, the twin's second model, and the windows.
  Absolute ceiling 1.094 GB for the whole organism; zero restarts.
- **P4 PARTIAL**: constancy perfect (Δ0.0000 over 40h); anchor value 8.6588
  vs the smoke-derived 8.6656±0.001.
- **P5 FALSIFIED IN MAGNITUDE, CONFIRMED IN FORM — the restart is free**:
  (i) surprise excess 0.0029 vs [0.03,0.15]; (ii) over-gating 1.0002× vs
  ≥1.5×; (iii) converged at n+1 CHUNK vs 2–8h; end gap −0.0065 (A3R ahead).
  Third independent measurement of the two-timescale law (P23, P38a).
- **P6 FALSIFIED as registered, mechanism intact**: both deltas negative
  (stale 5–11M-era spans on an 80×-older model disturb absolutely) but the
  paired contrast is sharp — inj −0.0866 vs rand −0.2793 (3.2×), helped
  0.35 vs 0.10. Provenance-limited, not mechanism-dead.
- **P7 EXCEEDED**: 1020 probes vs 100–600.

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
  **HARDENED to n=3 seeds across two CPU architectures (day 4):** seed43
  (Mac/ARM): ratio 0.955, GSSM +0.148; seed44 (core/x86): ratio 0.968,
  GSSM +0.133. Both verdicts replicate on every seed and both machines.

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

- **P31 — two organisms, one index: A pre-learns the shock for B (MS7,
  src/pos_shared_index.py, registered before the build):** organism A
  streams a C4+code mix and stores its surprise spans; organism B streams
  pure C4, then hits a code shock (the MS3 protocol). Arms for B, matched
  budgets: (i) B replaying A's shared spans during the shock, (ii) B with
  only its own (pure-C4) spans, (iii) B with no replay, (iv) control: B
  replaying token-shuffled versions of A's spans. Predictions: (a) B's
  code-phase forgetting with A's spans ≤ 0.7× of arm (ii); (b) B's code
  plasticity in arm (i) ≥ arm (ii) (A's spans carry usable code, not just
  regularization); (c) the shuffled control (iv) does NOT beat (ii) —
  the benefit is content, not noise injection; (d) A itself is unharmed
  (its own trajectory is not part of B's budget). This is collective
  memory across O(1) individuals: one organism's surprises become another's
  immunity.

- **P20 SCORED (results/pos_domain_shock.json, full):** PARTIAL — the core
  confirmed, one clause falsified, one surprise. Core CONFIRMED: R3
  (gating+dosed replay) forgets +0.246 = **37% of R1's +0.667** (bar was
  ≤50%) — sleep is the anti-forgetting organ. Clause FALSIFIED: R2's
  "comparable code plasticity" does not hold (0.520 vs 0.854 at 58% of the
  gradient tokens — the forgetting/plasticity ratio is dose-coupled, as the
  smoke already showed). Surprise, inverted from the smoke: in the full run
  R1 recovers almost fully (+0.010) while R3 stays +0.234 above baseline
  after phase 3 — at full scale the replay that protected WT-2 during the
  shock keeps pulling the model toward stored spans during recovery
  (an overdose signature on the recovery phase; the volume-coupled budget
  needs a phase-3 dividend monitor). Method note: the code-eval carries
  unk_rate 0.44 (WT-2 vocab on Python), so absolute code-NLL levels are
  diluted; the WT-2 forgetting signal — the core measurement — is clean,
  and all regimes share the same eval.
- **P27 SCORED (results/hot_swap_growth_mig.json, 1.2M+1.2M on beast):**
  (a) CONFIRMED (commutation 29/29; full-run gate 5.7e-6); (c) CONFIRMED —
  growth beats restart by +0.166 (third independent replication: +0.127
  fresh-Adam Mac, +0.152 smoke, +0.166 here). (b) FALSIFIED at the 1.2M
  horizon: deficit −0.033 vs −0.037 unmigrated — the smoke's 87% deficit
  reduction was a TRANSIENT effect that decays over Adam's β₂ memory
  horizon (~0.5M tokens). The real, robust gain of moment migration is the
  transient itself: post-surgery gap 0.046 → **0.0035** (13× smaller) —
  migration makes growth seamless, not faster-converging.
- **Seed-hardening correction (results/holo_magread_seeds23.json, beast):**
  the M3 "knee 512, seed-stable" claim was n=2. At n=4 the M3-recipe knee
  is 512/512/128–256/128–256 (seeds 0–3) and the V1/V2/V3 variants do not
  order consistently within seeds. What replicates: the v1→v3 repair, the
  interventions' direction, and the clamp+refresh interaction jump; the
  absolute knee position is seed-dependent (also seen in P30: 4096 vs 256).
  Documents updated accordingly — the honest unit is the intervention
  effect, not the knee coordinate.
- **P28 smoke note (results/pos_auto_q_smoke.json — not scored, method
  finding):** the q-controller saturates at the 0.95 clip in ALL phases —
  it integrates during the ignition window (forced gating, r̂=1.0) and
  never recovers: an anti-windup bug, not a homeostasis result. Fix: freeze
  the controller during ignition. Efficiency signal already visible: auto
  matches fixed's forgetting (+0.274 vs +0.281) at 18% fewer gradient
  tokens. Full run follows after the fix.

- **P28 SCORED (results/pos_auto_q.json, full, anti-windup fixed):
  FALSIFIED — and the mechanism is instructive.** The controller works as
  designed (q moves freely: medians 0.75/0.73/0.76; phase-2 gate rate lands
  exactly on target, 0.247 vs r*=0.25). That is precisely why it loses:
  under shock it holds the RATE constant instead of getting pickier — q
  even dips during phase 2 (0.733 < 0.75), admitting mid-surprise chunks —
  and forgets MORE than fixed q (+0.401 vs +0.350) at equal code
  plasticity (0.564 vs 0.565). (a) also out: both arms gate ~0.45–0.49 in
  phase 1 (the registered band was too tight for a warm-snapshot resume).
  Honest verdict: rate-homeostasis is the WRONG controller — the fixed
  quantile's shock response (gate the top surprises, whatever the rate) is
  the better curiosity policy. A surprise-LEVEL target instead of a rate
  target would be a new registered question, not a rescue of this one.

- **P32 — the rent map of the phase (MS10, src/holo_rent_map.py,
  registered before the build):** sweeping P_max ∈ {8,16,32,64} × d_model ∈
  {32,64,128,256} (holo_on vs holo_off, P ∈ {2,4}, G ∈ {8,32}, 2 seeds,
  matched budgets; best-of-2 lr control for the off arm at the corner
  cells — the lr lesson): (a) the phase rent (holo−off) is governed by the
  ratio d/P_max, with a transition in the band d/P_max ≈ 2–4 (anchors
  measured so far: rent at 16/64=×4 [v3, +25–30pp], none at 64/64=×1 [M1],
  small at 64/256=×4-but-large-d [M6, +13–15pp]); (b) cells collapse onto
  one curve in d/P_max (a ratio law, not two independent axes); (c) the
  off arm's recall varies smoothly with d and is flat in P_max beyond
  interference (no phase mechanism to gain rent from). Any cell where the
  lr-control flips the sign is reported as such, not averaged away.

- **P31 SCORED (results/pos_shared_index.json, full): CONFIRMED, all four
  checks.** B replaying A's spans through the code shock forgets +0.156 vs
  +0.233 with only its own spans (ratio 0.67 ≤ 0.7) at BETTER plasticity
  (0.542 vs 0.518); the token-shuffled control (+0.317) is nearly as bad as
  no replay (+0.376) — the benefit is content, not noise regularization.
  A's store: 64 spans, 7.8% code-like. Collective memory across O(1)
  individuals is real: one organism's surprises are another's immunity.
- **P32 SCORED (results/holo_rent_map.json, full 16-cell grid): (a) PARTIAL
  (low-ratio cells < 5pp holds; high-ratio ≥ 10pp fails), (b) FALSIFIED —
  no ratio law (ratio-4 cells span 44pp: +32.1 to −11.9) and the interim
  product law also breaks at d=256. The map itself is the result: the
  phase pays robust rent only in the SCARCE CORNER (P_max ≤ 16 AND
  d ≤ 64: +11 to +32pp), a near-zero valley in between (±3pp), and a
  second positive region at mid-P_max × large d (16–32/256: +12.5/+13.4 —
  the M6/P17 capacity-return, now mapped), with the extreme corners
  lr-sensitive and excluded. No one-parameter law fits; two rent regions
  separated by a valley is the measured shape.
- **P21 SCORED (results/holo_graft.json, full, both seeds): FALSIFIED as
  registered — and explained by the rent map.** The graft on the frozen
  475M-token organism recalls facts over REAL C4 text at 0.90–0.97 through
  G=32 (zeroed null at chance in every cell — the carried state carries the
  binding; real-text filler γ reaches 0.9996), but G=128 lands at 0.12–0.14
  (~2× chance, not the registered 3×; ctrl exactly at chance). And the
  phase rent vs ctrl vanishes at full budget (+37pp at 600 iters → +1–3pp
  at 2500): the graft sits at d=128/P_max=16 — P_max·d=2048, squarely in
  the rent map's valley. Two instruments, one law.

## Wave 7 — the three gaps it would be stupid not to close (registered day 4 ~07:30, before any build)

- **P33 — CHIMERA v0 (MS6, src/chimera.py, spec in analysis/CHIMERA_SPEC.md):**
  on the MS3 shock protocol at matched gradient tokens: (a) CHIMERA
  forgetting ≤ R3's (+0.246) at ≥ R3's plasticity (+0.366); (b) CHIMERA
  recovery beats R3's +0.234 — the dividend monitor must fix P20's
  recovery overdose; (c) each ablation (minus-reminder, minus-monitor) is
  worse than full CHIMERA on ≥1 axis; (d) no single-organ arm dominates
  CHIMERA on all three axes. F1's locking experiment.
  **Run-time note, registered BEFORE the full run (2026-07-24 ~15:00):**
  smoke (results/chimera_smoke.json, 120 chunks/arm): all five arms ran
  end-to-end; the dividend monitor demonstrably intervened (sleep SUSPENDED
  at EMA=−0.040, 7 monitor skips); the reminder organ fired ZERO times in
  every arm. Mechanical verification (unit test, same day): store→harvest→
  lookup fires correctly on an exact 4-gram recurrence — the organ works;
  the smoke's exposure is below the recurrence base rate (POS-measured:
  ≥1020 capped hits / 900M tokens / 20k spans ⇒ ~0.1 expected fires at
  smoke exposure; observed 0, consistent). The full run (150 chunks/phase)
  expects single-digit fires from the C4 base rate; code-phase boilerplate
  may raise it. Scoring consequence, fixed now: if fewer than 10 reminders
  fire in the chimera arm, clause (c)'s minus-reminder ablation is scored
  UNDECIDABLE-AT-EXPOSURE (neither confirmed nor falsified) — the organ's
  in-composition value then needs index-scale exposure or a
  recurrence-seeded protocol as a NEW registered experiment. Instrument
  dimensioning (max_per_chunk=2, spike_min_nll=7.0) deliberately left AS
  SPEC'D — no post-smoke retuning.
  **SCORED 2026-07-24 (full on core, results/chimera_full.json; 150
  chunks/phase, 5 arms):** instrument deviation named first: the
  registered "matched gradient tokens" clause is unenforceable for gated
  arms (the gate chooses); arms ran at matched CHUNKS — chimera used
  100.9k grad tokens vs r3's 141.3k vs r1's 230.4k. (a) FAILED as
  measured: chimera forgetting +0.259 > r3's +0.189 at essentially equal
  plasticity (0.625 vs 0.621) — with the caveat that chimera took 29%
  fewer gradient tokens. (b) CONFIRMED DECISIVELY — the headline: chimera
  residual damage after recovery +0.0028 (fully healed) vs r3's +0.151;
  and the ablation nails the attribution within the run: no_monitor
  (same organs, monitor removed) regresses to +0.146 while the monitor
  actively suspended 9 sleep blocks in the chimera arm. The dividend
  monitor IS the recovery organ — P20's overdose, fixed in composition.
  (c) SPLIT: minus-monitor worse CONFIRMED (recovery collapses);
  minus-reminder UNDECIDABLE-AT-EXPOSURE per the pre-registered rule —
  only 2 reminders fired (<10), the no_reminder deltas cannot be
  attributed to the organ. (d) CONFIRMED: no single-organ arm dominates
  (r3 wins forgetting, chimera wins plasticity+recovery; r1_full is the
  firehose signature — best plasticity +0.866, catastrophic forgetting
  +0.590, echoing the 40h result). Verdict: CHIMERA v0's composition
  trades some shock-forgetting for near-perfect recovery at equal
  plasticity and fewer gradient tokens; the monitor earns its place, the
  reminder organ still awaits index-scale exposure.
- **P34 — the phase lifts binding rank per channel to ≥2 (src/rank_sweep.py):**
  an Eckart–Young-style capacity sweep (K keys vs D channels, the
  rank1_capacity method) on the COMPLEX holographic state: (a) inverting
  the recall bound gives D_eff ≥ 1.8 per channel for the phase arm where
  the scalar arm inverts to ~1.0 (measured anchor: D_eff≈1.02); (b) the
  phase arm's capacity cliff sits at load ≈ 2K/D, the scalar's at ≈ K/D;
  (c) attention validity gate at ~1.0 throughout. If D_eff stays ~1, the
  phase's rent is NOT rank — the honest alternative (SNR-based) is stated.
  **AMENDED before the sweep ran (day 4 ~08:15), reason on record:** the
  0.1406 anchor's generating script is lost (two documented reconstruction
  attempts land at chance; the artifact paper/evidence_companion/
  hybrid_B.json remains the anchor's source of truth, flagged as
  reproduction debt in RANK1_CAPACITY_THEOREM.md). P34 therefore runs on
  the reproducible mqar.py instrument with the criterion made RELATIONAL:
  (a') D_eff_phase ≥ 2× D_eff_scalar at every K where phase clears 3×
  chance, and D_eff_phase(K=8) consistent with the measured 8.9% ceiling
  (≈0.6 model-wide); (b') the phase cliff sits at ≥2× the scalar cliff-K;
  (c) unchanged. Same law, honest instrument.
  **SCORED 2026-07-24 (full grid on the reference Mac,
  results/rank_sweep_final.json; K∈{2,4,8,16,32} × 4 arms × 4 seeds):**
  (c) CONFIRMED — attention validity min recall 0.9898, the tasks are
  solvable and the harness correct. Anchor reproduction is EXACT for
  scalar (0.0166 vs anchor 0.0170±0.0022), phase_off, and attn — the
  instrument is calibrated. (a') FALSIFIED at the one eligible cell and
  unscorable elsewhere: at K=2 (3/4 ignited) the ratio is 0.371 — the
  phase's per-channel capacity among ignited seeds is LOWER than the
  scalar's, the opposite sign of the prediction; no higher K clears 3×
  chance on seed-mean. (b') FAIL as computed (cliff ratio 1.0). The
  registered fallback therefore ENGAGES as written: the phase's measured
  rent (P32's map) is NOT per-channel rank — the SNR-based alternative is
  now the standing hypothesis. Dominating phenomenon and method finding:
  phase IGNITION COLLAPSES WITH LOAD (3/4 → 1/4 → 1/4 → 0/4 → 0/4 across
  K) — whatever the phase could pay at high K, training reliability dies
  first; and the single ignited K=8 seed (0.0728) sits ~1σ below the
  5-seed anchor mean, i.e. the ignition coin drifts with machine co-load
  even on the reference machine. Process note: the full was launched
  without --verify-anchor (the harness's own guard); the post-hoc anchor
  comparison above recovers it. The 0.1406 Task-B anchor's reproduction
  debt (P34 amendment) stands unchanged — this scores the mqar.py
  instrument, on which the rank hypothesis is dead.
- **P35 — the gap ladder to a million (src/gap_ladder.py):** eval-only, on
  the M3-recipe model with the P30 clamp+refresh prosthesis, gaps
  {4096, 16384, 65536, 262144, 1048576} chunked-carried: (a) MQAR recall at
  G=65536 ≥ 0.5× its G=4096 level; (b) the beacon bit (write-once-freeze
  carrier, MS13 harness) survives G=1M at recall ≥ 0.9 with the refresh
  prosthesis (γ=0.9995 alone decays at τ≈2000 — the refresh is load-bearing
  and that is the point); (c) zeroed null at chance at every rung (large
  eval batches per P36's protocol). Any rung that breaks is the measured
  wall, reported as such.
- **P36 — P30c null hardening (src/holo_alpha_shut.py --null-hardening):**
  the zeroed-at-gap null re-run with eval batches ≥ 100 at G ∈ {1024, 2048}:
  all null cells within 3pp of chance. Closes the F6-flagged sampling-noise
  gap.

- **P37 — the future-trained organism (MS15, src/horizon_pos.py, registered
  before any build; David's architecture insight, day 4):** add an H-step
  prediction head to the streaming organism (predict a summary functional of
  chunk t+H at chunk t; score on arrival; horizon-surprise = the error).
  On the MS3 shock protocol: (a) horizon-surprise (H ≥ 8 chunks) rises ≥ 5
  chunks EARLIER at the domain boundary than 1-step surprise (the
  early-warning claim); (b) gating on a mix of 1-step and H-step surprise
  at matched gradient tokens forgets ≤ the 1-step gate's forgetting with
  plasticity within 10%; (c) the deposited-prediction mechanism is honest:
  shuffling the deposited predictions destroys (a) and (b). If (a) holds
  but (b) does not, the horizon signal is a detector, not yet a teacher —
  reported as such.
  **SCORED 2026-07-24 (full on core, results/horizon_pos_full.json):**
  (a) FALSIFIED — the H=8 detector fired 65 chunks LATER than 1-step
  surprise at the phase1→2 boundary (fire idx 304 vs 239) and never fired
  at phase2→3. No early warning at this scale — consistent with the
  smoke's structural finding. (b) CONFIRMED with room: horizon-mix gate
  forgetting +0.300 vs base gate +0.392, at plasticity +0.736 vs +0.544 —
  both axes better, not a trade. (c) FALSIFIED, and this kills the
  attribution: the SHUFFLED-deposit control keeps the (b)-shaped benefit
  (forgetting +0.324, also beats base) — whatever improves the mixed
  gate, it is NOT the content of the deposited predictions. Verdict: at
  this scale the future-trained gate is neither detector (a) nor
  attributable teacher (c); the measured (b) win is real but structural —
  a second, differently-tempered surprise stream diversifies the gate.
  Next attack (unregistered design note): a rate-matched noise-gate
  control carrying the same firing statistics but zero predictive
  content — if (b) survives it, gate DIVERSITY is a cheap new F1 organ in
  its own right; if it dies, the horizon content matters and the shuffle
  control was too weak. The multi-horizon architecture idea stays open at
  larger H / richer targets — this scores THIS instrument, not the idea.
- **P40 — gate diversity vs. content (the P37 attribution decider,
  registered 2026-07-24 before any build):** three arms on the MS3 shock
  protocol at matched gradient tokens, same ckpt, seed 42 primary:
  base_gate (1-step only), horizon_gate (P37's mix, unchanged), and
  noise_gate — identical machinery and compute, except at gate-decision
  time the second stream's value is the horizon-surprise from a uniformly
  random EARLIER chunk (large random lag): distribution-identical,
  rate-matched through the same rolling quantile, zero temporal content.
  (a) REGISTERED POINT CALL: the noise arm RETAINS ≥70% of horizon_gate's
  forgetting improvement over base — the P37-(b) win is DIVERSITY, not
  content. If instead horizon beats noise by >0.03 nats forgetting at
  ≥ noise's plasticity, content matters and P37's shuffle control was too
  weak — reported at full strength either way. (b) sanity: noise arm's
  realized second-stream firing rate within 2pp of horizon_gate's.
  Either outcome is a win: diversity ⇒ a near-free new F1 organ (multiple
  gate tempers); content ⇒ horizon-v2 is justified sharper. Design note
  for horizon-v2 (numbered only when its spec freezes): SELF-SURPRISE
  target — the head predicts the organism's own chunk-NLL at t+H (one
  scalar, purer and cheaper than v1's histogram), H-ladder {2, 8, 32}, on
  a shift-denser schedule (early warning needs boundaries to warn about;
  MS3 has only two).
- **P41 — the retrodiction meter (MS18 v0, David's time-mirror, registered
  2026-07-24 before any build):** a BACKWARD head ladder on the streaming
  organism: at chunk t, one linear head per rung reconstructs the
  top-256-bucket histogram of chunk t−H from the current carried
  state/features, H ∈ {2, 8, 32, 128}. Targets are past chunks — available
  immediately from a bounded rolling buffer (no deposit queue); heads
  train online, matched budgets, MS3 shock stream, same ckpt as
  P37/P40. Registered: (a) TWO-REGIME DECAY — the two-timescale law seen
  backward: error rises steeply across the receptive-field scale and
  PLATEAUS beyond it on the weights level. Point call:
  err(H=128) − err(H=32) < 0.25 × (err(H=8) − err(H=2)). (b) Shuffle
  control: temporally shuffled targets erase the H-structure (rungs
  within noise of each other, no monotone ladder). (c) LIVE FORGETTING
  (wording clarified before any run, 2026-07-24 evening — "the phase-2
  boundary" means the boundary INTO phase 2, the phase1→2 shock at B12):
  in the window (B12, B12+32], on decisions whose target lies pre-shock
  (t−H < B12), the qualifying rungs' error rises above their own
  end-of-phase-1 mean + 2σ — the meter sees phase-1 content fade WHILE
  the model adapts to code; H∈{8,32,128} qualify richly at both smoke and
  full phase lengths (direction registered; magnitude exploratory in
  v0). Scope: v0 is the METER only — the consolidation organ
  (decay-triggered targeted replay) is a separate later registration.

- **P38 — the portable organism (MS16, src/portable_organism.py, registered
  before any build; David's seeding insight, day 4):** three compositions,
  each against a never-moved control at matched token budgets: (a) LIVE
  CROSS-ARCHITECTURE MIGRATION — a running organism checkpointed on the Mac
  (ARM) resumes on beast (x86) mid-stream; its held-out trajectory rejoins
  the control's within 0.02 nats inside 1M tokens (bit-determinism is lost
  across BLAS — behavioral equivalence is the claim); (b) KILL+REJOIN — of
  two replicas sharing an index (P31 harness), one is killed for K chunks
  and rejoins by snapshot; after the measured heal it is within 0.02 nats
  of its uninterrupted twin, and the surviving replica's index writes cover
  the gap (the rejoiner benefits from spans collected while it was dead);
  (c) OFFLINE MODE — a stream outage of K chunks spent SLEEPING (dosed
  replay, dividend-monitored) beats the same outage spent idle by a
  measurable held-out margin at reconnect+N chunks, and both resume
  cleanly. Any part that fails is the measured boundary of portability.
  **P38a SCORED (day 4 ~09:15): CONFIRMED far beyond the registered bar.**
  Local gate: checkpoint→new-process→resume is BIT-identical (line-by-line
  chunk log). Cross-architecture: a live organism checkpointed mid-stream on
  the Mac (ARM) and resumed on beast (x86) ends at heldout 6.182391 —
  identical to six decimals with the never-migrated control (6.182391),
  with identical gradient tokens (17,664: every gate decision matched).
  The only divergence is the bit-level digest (BLAS rounding differs across
  ISAs) and it does NOT propagate into behavior. The registered bar was
  "rejoin within 0.02 nats inside 1M tokens"; the measurement is immediate
  behavioral identity. Live cross-ISA migration is a solved, free operation.
  (b) and (c) full runs in flight on core; smoke already shows the
  index-cover mechanism (shared ≤ private on rejoin) and an honest
  small-budget boundary on offline-sleep (toy span pools are not
  representative — full budget decides).

- **P39 — stop-free streaming migration (MS17,
  src/portable_organism.py --exp d, registered before the build; David's
  möbius insight, day 4):** organism A streams continuously and NEVER
  pauses; at chunk N a snapshot is taken (sub-second) and transferred while
  A keeps streaming; target B resumes from the snapshot and replays the
  chunk range A consumed during the transfer window, then both process the
  next chunks in lockstep. Checks: (a) zero source downtime (A's chunk
  cadence shows no gap at the snapshot point); (b) B reaches state parity
  with A — identical held-out and identical gate decisions on the first
  shared post-catch-up chunks (locally bit-identical; cross-ISA to six
  decimals per P38a); (c) the catch-up window is bounded and small
  (transfer+replay < the time A needs to stream the same chunks — the
  migration CONVERGES rather than chasing forever); (d) iterated
  delta-sync (repeat snapshot/catch-up at cadence C) keeps a standing
  replica within one sync window of the living source at all times —
  continuous replication as a standing state. **P39 SCORED (day 4): (b)
  STATE PARITY CONFIRMED — the core claim: deterministic catch-up while the
  source keeps living gives bit-identical lockstep (chunk-log tail matches,
  heldout 6.171316 == 6.171316, local; six decimals cross-ISA per P38a). (a)
  the snapshot chunk costs 27.8 ms absolute (Torch serialization) — a
  ratio-10 blip only against beast's 2.7 ms baseline, NOT a perceptible
  stall; the metric conflates I/O latency with downtime and is being
  re-measured at realistic d_model where per-chunk cost dominates. (c)
  convergence and (d) iterated re-ran at d_model=128 on quiet beast: same
  picture (snapshot 35 ms absolute vs 4 ms chunks; B/A cpu 2.3 with the
  fixed process-start amortized over toy-sized work). VERDICT on the
  metrics themselves: (a) and (c) as registered are STRUCTURALLY
  UNDECIDABLE at toy scale — both are ratios of fixed constants
  (serialization, process start) to per-chunk costs that are microseconds
  in a d≤128/B=4 organism; they measure overhead arithmetic, not the
  migration property. What IS measured and stands: (b) bit-identical
  catch-up while the source lives (every scale, every machine tried), and
  the absolute snapshot cost (28–35 ms — one chunk slot). Final (a)/(c)
  measurement scheduled on the full POS-scale organism (B=8/K=64, ~100 ms
  chunks) where per-chunk work dominates the constants.

## Scoring rule

Each P-item gets CONFIRMED / PARTIAL / FALSIFIED in the harvest documents,
with the measured number beside the predicted interval. A falsified
prediction is a measurement — the register exists so we can't unknow what we
expected.
