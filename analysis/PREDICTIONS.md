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

## Scoring rule

Each P-item gets CONFIRMED / PARTIAL / FALSIFIED in the harvest documents,
with the measured number beside the predicted interval. A falsified
prediction is a measurement — the register exists so we can't unknow what we
expected.
