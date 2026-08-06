# RETRO_BUILD_NOTES — P61 / MS-E build findings

Format and deviation notes for `src/retro_keyed_run.py` (P61 retrodiction
organ, keyed v1). The spec is `analysis/RETRO_SPEC_DRAFT.md`; the three open
questions were frozen in the registration (Q1 = both substrates in role
division, Q2 = span prefix as value, Q3 = deferred). Everything below is a real
finding from the build — flagged, NOT silently absorbed, per the build brief.
The lead should read the FOUR deviations before the full and amend the register
where they change the registered wording.

## Smoke result (nice -n 19, 16.5s, well under the 5-min cap)

    MQAR instrument arm:  train_acc 1.0   ladder contrast [0.50 (H=2), 0.18 (H=8)]
    ORGANIC arm:          train_acc 0.0   ladder contrast [0.00, -0.02]
    clauses: instrument_ok=True  a_pass=False  b_pass=None  c_pass=False  d_pass=None

The instrument gate OPENS (contrast 0.50/0.18 vs a zeroed-null at ~0.06 — the
keyed read works and its retention decays with lag). The organic arm shows no
signal on this checkpoint's store (see DEVIATION 3). Field structure, both
controls, the self-referenced trigger, and all `p_retro_<letter>_pass` fields
are exercised. `results/retro_keyed.json` carries the cadence block d128/B8/K64.

## DEVIATION 1 — the fork checkpoint is a SCALAR model, not a holographic one

`results/pos_snapshots/ckpt_359050240.pt` is a `StreamingNoPELM` snapshot: its
`arms.A3.model` state_dict is `W_v / W_gamma / W_alpha / W_out` — the real
scalar GSSM-Selective scan (use_phase=False world). It carries NO holographic
key channel (`W_key`, complex state, de-rotation). The spec §1.3 read
`read = Re(Z e^{-i phi_q})` does not exist on this model.

Resolution (documented, needs a register amendment): the keyed read is measured
on the holographic F3 stack (`holo_stream_recall._build_lm`, use_phase=True,
separate_qk=True) at d128/K64, NOT on the POS model's own state. The spec's
wording "use the organism's OWN native keyed read" is therefore not literally
executable against this checkpoint. Two clean paths for the full, lead to pick:
  (a) run the whole organ on a holographic-POS organism (a POS run whose scan
      is the holographic layer with the phase channel on) — then the read IS
      native; or
  (b) keep the keyed read as a parallel holographic channel over the same
      WT-2 stream, seeded by the POS store (what the harness does now).
Path (a) is the truer form of the spec; it needs a holographic-POS checkpoint
that does not exist yet.

## DEVIATION 2 — a trained key channel is a PRECONDITION (untrained W_key is null)

At init `W_key` is deliberately small (phi≈0 → the read sits in the
Selective/real-write regime, byte-identical to use_phase=False). An UNTRAINED
holographic layer therefore cannot discriminate keys: measured contrast ~0.0
and a slightly NEGATIVE phi-margin. Keyed retention only exists AFTER the key
channel is trained on a recall objective. Consequence for the clauses: (a) and
(b) require a real training budget; they cannot be read off an untrained state.
The smoke trains 1200 iters to open the gate; the full needs more (see costs).
This is why the meter clauses are gated on `instrument_ok` — see DEVIATION 4.

## DEVIATION 3 — this checkpoint's store is too id-sparse for the organic arm

The organic arm draws its key/value pools from the POS checkpoint's
`index_state.keys`. On `ckpt_359050240.pt` that store yields too few DISTINCT
token ids to build a well-posed remapped recall task (the harness falls back to
the stream buffers and still degenerates: organic train_acc 0.0). The MQAR
instrument arm is unaffected (it uses synthetic disjoint ranges). For the full,
the organic arm needs EITHER a checkpoint with a richer harvested store, OR the
organic keys drawn from a live harvest over the WT-2 stream at run time rather
than from the frozen `index_state`. Recommend the latter: harvest organic
spikes live (the chimera `KeyedSpanStore.harvest` path) so the value = span
PREFIX (Q2) is a real streamed span, not a remapped id. This also makes Q2
literal, which the current remap only approximates.

Note on the remap: because the F3 recall harness addresses key/value/filler by
CONTIGUOUS DISJOINT id ranges (`_gap_vocab`), organic WT-2 ids (not disjoint)
are remapped onto fresh contiguous ranges sized by the store. The store ids
thus seed the pool SIZES, not the literal token identities. A live-harvest
organic arm would remove this remap.

## DEVIATION 4 — the v0 discipline is enforced: no vacuous pass on a dead ladder

P41 v0's lesson ("a formal pass on a flat-zero contrast is vacuous and NOT
claimed") is coded in: if the instrument gate is shut (MQAR contrast below bar),
clauses (a), (b), (c) return `None` (not-scorable), NOT a meaningless `True`.
A flat-zero ladder trivially satisfies "far drop ≥ 2× near drop" and "null ≤
0.02" — exactly the trap that killed v0 — so those passes are suppressed unless
there is real signal to measure. `p_retro_a_scorable` records the gate state.

## Task-size calibration (a measured cost finding, not a guess)

Keyed MQAR recall is unlearnable at large value/filler ranges: v_max=128/f=256
never trained (contrast ~0.0 at 800 iters); the F3-PROVEN sizes v_max=16/f=16
hit recall 1.0 in ~16s at d128/K64. The harness uses v_max=16/f=16.
**Cadence d128/K64 is NOT the limiter** — measured, it reaches full recall
exactly as the F3 default d64/K16 does (contrast +0.945 at both). Only the task
ranges mattered. P=1 opens the gate in the smoke budget; P=2 needs more iters
(smoke=P1, full=P2 by default).

## What the full needs (chunks/arms, no wall-clock guessing)

Per spec §5, budget-neutral at d128/K64. The build makes the eval cost concrete:
- **Training to open the gate:** ~1200 iters/arm opens P=1 (contrast ~0.45);
  P=2 and a clean organic read want the full ~6000 iters/arm (spec §5).
- **The eval cost driver is the UPPER ladder rungs**, not training: H=128 is a
  G=8192-token eval — 10.8s at NB=50, so ~86s at NB=400; H=32 (G=2048) ~2.8s at
  NB=50. Four rungs × {matched, zeroed} × {MQAR, organic} at NB=400 is the full's
  dominant cost. Order: tens of minutes per configuration, single runner —
  consistent with the spec's ~9-run estimate, and it fits a chain gap.
- **Actuator clause (d)** is a separate multi-run experiment (monitor / retro /
  retro_shuffled_trigger arms on the MS3 shock, spec §3.3) — left as documented
  full-run fields (`p_retro_d_*`), anchored to the dividend-monitor residual
  −0.040215 (chimera_v1.json).

## Clause field map (machine-checkable, per the scorer audit's readability rules)

Every clause writes a `p_retro_<letter>_pass` boolean (or `None` when not
scorable) beside its raw numbers; the artifact carries the cadence block. So the
auto-scorer (`src/score_predictions_v2.py`) can check P61 mechanically from day 1.
  (a) p_retro_a_pass  + a_contrast_H8/H32, a_instrument_ok, a_scorable
  (b) p_retro_b_pass  + b_ladder_contrast, b_far_drop, b_near_drop
  (c) p_retro_c_pass  + c_zeroed_null_median, c_bar
  (d) p_retro_d_pass  + d_monitor_residual_anchor, d_retro_residual (full-run)
