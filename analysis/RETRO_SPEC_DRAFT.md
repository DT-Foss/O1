# RETRO_SPEC_DRAFT — MS-E: the retrodiction organ, keyed v1

Design draft for the consolidation organ that P41 v0 licensed and could not
build. NOT a registration — the lead freezes P-numbers after review. This
document specifies the organ, the instrument, the loop, draft clauses with
falsifiers, the cost, and the open questions. Numbers cited from artifacts are
marked with their source; everything forward-looking is a design target, not a
measurement.

---

## 0. What we are building and why the v0 died

**The insight (MS18, David's time-mirror).** Forward surprise says LEARN NOW;
backward surprise says CONSOLIDATE NOW. A head that reconstructs something about
chunk *t−H* from the current state measures not foresight but RETENTION — an
online, per-chunk forgetting meter. The backward H-ladder {2, 8, 32, 128} is
the F3 persistence knee turned into a live observable instead of an eval probe.
The organ it licenses: consolidation triggered by MEASURED decay — when a
stored binding's retrodiction fails, replay exactly that binding. This is the
**precision version of the dividend monitor** (P54's measured recovery hero):
the monitor gates sleep on an aggregate held-out EMA and samples spans by a
freshness *proxy* (~1/age); the retrodiction organ replaces the age proxy with
a *measured* per-binding decay signal and replays the specific bindings that
decayed.

**Why v0 read null (P41, results/retro_pos_full.json + its register block).**
v0 asked a set of linear heads to reconstruct the **top-256-bucket histogram**
of chunk t−H from mean-pooled features. Scored on the REAL−SHUFFLED contrast
(the amendment that cancels the per-rung step-count confound), the per-rung
contrasts at end-of-phase-1 were +0.030/+0.059/+0.021/+0.021 (from the P41
scored register block; the JSON `p41_scoring.a_two_regime` carries the absolute
per-rung errors behind them — 4.346/4.317/4.320/4.408, err128−err32 = 0.0876),
each within ~1–2σ of zero, no ladder. The diagnosis, verbatim from the register:
*"chunk-specific retention is INVISIBLE to bulk-histogram reconstruction from
mean-pooled features on stationary text — the state's marginal statistics are
interchangeable across recent chunks, so the probe measures the marginal, not
memory."* And the standing v1 design note:
*"the F3 knee IS keyed retrodiction; the meter's v1 must therefore probe with
KEYS, not bulk summaries."*

**The one design decision that follows.** The v1 signal must be a **keyed
read**, not a learned reconstruction of a marginal. The organism already has a
keyed-read operation — the complex holographic write/read
(src/holo_stream_recall.py): a value is bound to a key by phase
`phi_w = phase_scale·tanh(W_key(x))`, written into the carried complex state
Z with γ-decay, and read back by de-rotating with the query key's angle,
`read = Re(Z · e^{-i·phi_q})`. **v1 measures the decay of this existing keyed
read as a live per-chunk observable.** It does not train a new marginal-reading
head — that is precisely the trap v0 fell into. The F3 knee (README §10) is the
same quantity measured in an eval; v1 measures it per chunk, per key, during the
stream.

---

## 1. The organ, precisely

### 1.1 Substrate and cadence
- Stack: the d128 holographic organism (src/holo_stream_recall.py's
  `HolographicGSSM` layer), same as the F3 / P30 / P35 line.
- Cadence block (recorded in the artifact, per the standing cadence-audit
  rule): **d_model 128, B 8, K 64, q 0.75, window 500** — the production
  cadence used by P54/chimera and the width curve, so the organ is measured on
  the same denominator as its baseline.
- Stream: the MS3 shock protocol (C4 → code → C4) at 1,000-chunk phases — the
  exact protocol on which the dividend monitor was measured (P54), so the
  baseline comparison is like-for-like.

### 1.2 What is "keyed", concretely
The keys are **not** invented for this probe; they are the keys the organism
already writes. Two candidate key sources, and the spec commits to the first
with the second as a registered fallback:

- **Primary — surprise-span keys (the store's own keys).** The chimera store
  already keys each harvested span by a content key
  (`store {key: (span, coord, surprise, age)}`, src/chimera.py). Each stored
  span was written into the carried state at its harvest chunk with a definite
  phase key. The retrodiction probe re-presents that key at a later chunk and
  measures whether the bound value still reads out. This makes retrodiction
  decay a property of the SAME objects the consolidation loop replays — signal
  and action are on one set of bindings, which is the whole point.
- **Fallback — synthetic MQAR keys injected on a schedule** (the F3 harness's
  disjoint key/value id ranges). Cleaner (exact ground-truth value) but
  off-distribution; registered as the fallback if the store's content keys are
  too collision-prone to read cleanly (open question Q1).

### 1.3 The retrodiction head / read
For a binding written at chunk *c* with key *k* and value *v*, and the current
chunk *t* with lag *H = t − c* (in chunks), the retrodiction **read** is the
native holographic read of the carried state Z_t de-rotated by k:

    v_hat(k, t) = Re( Z_t · e^{-i · phi_r(k)} )        [no new parameters]

and the **retrodiction error** is the reconstruction loss of v under that read,
scored at the value's token positions only (the F3 last-position convention):

    e(k, H) = NLL( v | v_hat(k, t) )      or      1 − recall(v | v_hat(k, t))

The **H-ladder** {2, 8, 32, 128} chunks spans the receptive-field scale
(~5–8 tokens = well under one 64-token chunk) up to well past it, so the ladder
crosses the two-timescale boundary that F3/P35 already located: the MQAR keyed
read holds to G≈16k tokens then walls (P35, results/gap_ladder_full), and the
γ=0.9995 carrier decays at τ≈2000 tokens without the refresh prosthesis (P35).
At K=64 tokens/chunk, H=128 chunks ≈ 8,192 tokens — inside the measured live
band, deliberately below the 16k MQAR wall so the ladder measures graded decay,
not a cliff. **This H-range is chosen from measured walls, not guessed.**

> Architecture sketch (d128 stack):
> ```
>   stream chunk x_t ──► HolographicGSSM layer ──► carried state Z_t (complex)
>                                                        │
>   store keys {k_i @ chunk c_i} ──► phi_r(k_i) ──► de-rotate Z_t ──► v_hat_i
>                                                        │
>                                          e(k_i, H_i) = NLL(v_i | v_hat_i)
>                                                        │
>                          ┌─────────────────────────────┴───────────┐
>                          ▼                                          ▼
>            RETENTION LADDER (the meter)                 CONSOLIDATION TRIGGER
>            e vs H, per key                              e(k_i) > θ ⇒ replay span k_i
> ```

The organ adds **no forward-path parameters** — it reads the state the forward
pass already produces. Its only cost is the extra keyed reads per probe
interval (Section 5).

---

## 2. The instrument: measuring retention without the v0 trap

The v0 failure mode is the design's primary adversary: a probe that reads the
MARGINAL (interchangeable across recent chunks) rather than the specific
BINDING. The instrument is built around two controls that separate "the read
sees the stored binding" from "the read sees a prior / a marginal". A clause
that cannot beat both controls is not measuring memory.

### 2.1 The two required controls
- **C1 — shuffled-key control (kills "reads a prior").** Present the WRONG key
  for each stored value: read Z_t with `phi_r(k')` for a k' drawn from other
  stored keys (matched count, same de-rotation machinery). If the true-key read
  and the shuffled-key read decay the same way, the read is not key-specific —
  it is reading the marginal, exactly as v0 did. The load-bearing quantity is
  the **REAL−SHUFFLED contrast** at each H, as the P41 amendment established.
  A ladder must appear in the CONTRAST, not in the absolute curve (the absolute
  curve confounds with the value's own frequency).
- **C2 — foreign-organism control (kills "reads a generic decay law").** Read a
  SECOND organism's carried state Z'_t (a different seed, same architecture,
  streaming the same protocol) with THIS organism's stored keys. The foreign
  state never wrote these bindings, so a true keyed read must fail on it at all
  H. If the foreign-state read succeeds, the "recall" is a property of the
  architecture's generic dynamics, not of what THIS organism stored. This is
  the stranger-verification logic (P60) turned inward: only the writer's own
  state should answer to the writer's keys.

### 2.2 The retention signal, defined for scoring
For each (key k, lag H): `retention(k, H) = e_shuffled(k, H) − e_true(k, H)`
(positive ⇒ the true key reads better than a wrong key ⇒ genuine binding). The
per-H retention is the median over the sampled keys. The two-regime (backward
two-timescale) shape is the registered structure: retention high and roughly
flat within the receptive-field/carrier band, then falling across the knee — the
F3 knee read backward.

---

## 3. The organ loop: measured decay triggers consolidation

### 3.1 Trigger
Every `probe_every` chunks, probe the store's bindings at their current lags.
A binding k qualifies for consolidation when its retention has decayed below a
threshold relative to its own freshly-written retention:

    retention(k, H_now) < τ_frac · retention(k, H≈2)          (per-binding, self-referenced)

Self-referencing to the binding's own H≈2 value (its retention when freshly
written) removes the value-frequency confound: each binding is compared to its
OWN best, not to a global constant. `τ_frac` is the one free threshold
(draft 0.5; swept in the run — Section 5).

### 3.2 Action
Replay exactly the qualifying spans (the store already holds `(key, span,
coord)`; replay slices identically, src/chimera.py). Budget: the SAME
budget-neutral carve-out the chimera arms use — consolidation chunks come OUT of
the wake block that triggered them (block = `--sleep-every`), so total chunks
visited equals every baseline arm's. This keeps the comparison at equal total
gradient exposure (the invariant P54 enforced).

### 3.3 What it is measured against — the baseline that counts
The comparison that decides the organ is **retrodiction-triggered replay vs the
dividend monitor** (P54's recovery hero), at equal budget, on the same MS3
shock:
- **Baseline arm = `monitor`**: the chimera dividend-monitored freshness sleep
  verbatim (EMA of held-out delta gates sleep; freshness ~1/age sampling).
  Its measured recovery residual is chimera's −0.040 (results/chimera_v1.json,
  p33/p54 scoring) — the number to match or beat.
- **Organ arm = `retro`**: identical stack and budget, but sleep is triggered
  by measured per-binding retrodiction decay and replays the decayed bindings.
- **Control arm = `retro_shuffled_trigger`**: trigger on the SHUFFLED-key decay
  signal (i.e. consolidate bindings the meter says are fine, chosen to match the
  count). If `retro` and `retro_shuffled_trigger` recover equally, the trigger
  carries no information — the organ is just extra replay, and the "measured
  decay" claim dies. This is the loop-level twin of C1.

---

## 4. Draft clauses (a)–(d), register-style, machine-checkable from day 1

Each clause names the artifact field that carries its verdict, so the auto-scorer
(src/score_predictions_v2.py) can check it mechanically. **Artifact design rule
(from the scorer audit's three readability findings):** every clause writes a
`pXX<letter>_pass` boolean AND the raw numbers beside it, and every cost-ratio
carries its (batch, chunk, d_model) cadence block. Draft artifact:
`results/retro_keyed.json`.

- **(a) THE KEYED METER READS SIGNAL WHERE THE BULK METER READ NULL.**
  The REAL−SHUFFLED retention contrast at the mid ladder (H=8 and H=32) is
  `≥ 0.10` (nats or recall points, instrument fixed below) and clears 2σ of its
  own shuffled-key null — i.e. the keyed probe sees the binding that the v0 bulk
  histogram could not. Fields: `p_retro_a_contrast_H8`, `..._H32`,
  `..._sigma`, `p_retro_a_pass`.
  *Falsifier:* contrast within 2σ of zero at every H ⇒ even keyed reads cannot
  see chunk-specific retention on this stream at this cadence, and the
  time-mirror meter is null on text (not just on bulk features) — a real
  negative that would send the organ to synthetic-MQAR substrate (Q1) or retire
  it. What each failure means is stated, not hidden.

- **(b) THE DECAY IS TWO-REGIME (the knee read backward).**
  Retention is high-and-flat within the carrier band and falls across the knee:
  `contrast(H=128) − contrast(H=32) ≥ 2 × (contrast(H=8) − contrast(H=2))` in
  magnitude (the decay accelerates past the receptive-field scale) — the
  backward image of the F3 forward knee (README §10) and consistent with the
  P35 walls (MQAR 16k, carrier τ≈2000). Fields: `p_retro_b_ladder[...]`,
  `p_retro_b_pass`.
  *Falsifier:* flat contrast across H (no ladder) while (a) passes ⇒ the read is
  key-specific but retention does not decay on the measured horizon — the meter
  works but there is no live forgetting to consolidate against at this scale;
  the organ has a signal but no job here, localise the horizon where it would.

- **(c) THE FOREIGN STATE DOES NOT ANSWER (specificity).**
  Reading a foreign organism's state with this organism's keys yields retention
  within 2σ of zero at ALL H (`median foreign contrast ≤ 0.02`), while the
  native read passes (a). Only the writer's own state answers its own keys.
  Fields: `p_retro_c_foreign_contrast`, `p_retro_c_pass`.
  *Falsifier:* foreign state answers (contrast ≥ the native's) ⇒ the "recall" is
  a generic property of the architecture's dynamics, not of stored content — the
  meter is measuring the operator, not the memory, and the keyed claim is as
  null as v0 was (different mechanism, same verdict). Named, not smuggled.

- **(d) MEASURED-DECAY CONSOLIDATION MATCHES OR BEATS THE DIVIDEND MONITOR.**
  On the MS3 shock at equal budget, the `retro` arm's recovery residual is
  `≤ monitor`'s (i.e. `≤ −0.040`, results/chimera_v1.json), AND the
  `retro_shuffled_trigger` control is strictly worse than `retro` by `≥ 0.03`
  (the trigger carries information). Fields: `p_retro_d_retro_residual`,
  `..._monitor_residual`, `..._shuffled_residual`, `p_retro_d_pass`.
  *Falsifier 1:* `retro` residual > `monitor` ⇒ the precision trigger does not
  beat the heuristic at this scale — the dividend monitor stays the recovery
  organ, and retrodiction is a meter (a/b/c) but not (yet) a better actuator;
  reported as such, no reframing.
  *Falsifier 2:* `retro ≈ retro_shuffled_trigger` ⇒ any recovery gain is from
  extra replay volume, not from WHICH bindings were replayed — the "measured
  decay triggers consolidation" thesis dies at the loop level even if the meter
  (a–c) is sound. This is the decisive control and is called out as such.

Scoring note (from the scorer audit): (a)–(c) are meter clauses read off logged
retention curves; (d) is the actuator clause read off the four arms' recovery
residuals. Keep the two separable — a sound meter with a null actuator is a
publishable split (it was for P37: detector-not-teacher), and the artifact must
let the scorer see which half passed.

---

## 5. Cost, in chunks and runs (no wall-clock guessing)

Cadence d128/B8/K64 throughout. Budget-neutral: consolidation chunks are carved
from wake blocks, so the ARM TOTALS below are wake+sleep summed and equal across
arms.

- **Meter development (a)–(c): one run, ~3,000 chunks.** One organism streams
  the MS3 protocol; the retention ladder + both controls (shuffled-key,
  foreign-state) are logged every `probe_every` chunks. The foreign-state
  control needs a SECOND organism streaming in parallel (same protocol, seed+1)
  whose state is read but never written by the probe — so the meter phase is
  **2 organisms × ~3,000 chunks**. No new training; the probe is extra reads.
- **Actuator (d): four arms × ~3,000 chunks** (`monitor`, `retro`,
  `retro_shuffled_trigger`, and a `no_sleep` floor for the residual anchor) on
  the 1,000-chunk-phase MS3 shock. Reuses the chimera harness's arm scaffolding
  and budget-neutral carve-out.
- **Threshold sweep:** `τ_frac ∈ {0.3, 0.5, 0.7}` on the `retro` arm only
  (3 × ~3,000 chunks), to locate the trigger operating point — registered as a
  sweep, not tuned post hoc.
- **Probe interval:** `probe_every` draft 25 chunks (matches the chimera
  reminder cadence); the per-probe cost is `n_store` keyed reads over the K=64
  value window — O(store size), a few thousand token-reads per probe, negligible
  against the forward pass.
- **Total order:** ~9 organism-runs of ~3,000 chunks = a minutes-to-low-tens-of-
  minutes chain on the d128 stack (same class as the chimera / pixel runs),
  fits a single chain-gap on one runner. It chains naturally AFTER P54's
  chimera arms (shares harness) and does not need a fresh long pretrain — it
  forks the same POS checkpoint the chimera arms fork.

---

## 6. Open design questions (marked as open — not guessed)

- **Q1 — read cleanliness of the store's content keys.** The store's content
  keys were written on real C4/code text, where keys are not the disjoint,
  collision-free ids the F3 MQAR harness uses. It is OPEN whether the native
  keyed read is clean enough on organic keys to give a readable
  REAL−SHUFFLED contrast, or whether collisions between similar spans blur it.
  The registered fallback (synthetic MQAR keys, §1.2) exists precisely for this,
  but choosing it trades ecological validity for a clean read — that trade is a
  decision for the lead, not a default I will pick. **This is the single
  highest-risk assumption in the design.**
- **Q2 — value target for the retrodiction loss.** For a stored span, is the
  "value" the whole span's tokens (rich but long, may exceed one carrier's
  clean read length — P35's read holds ~one chunk) or a compressed value token
  (clean read, but reintroduces a summary that could be marginal-readable like
  v0)? OPEN. Leaning span-tokens with the read scored only over the first
  ~K tokens after the key, but this needs the smoke to confirm the read length.
- **Q3 — is retrodiction error a BETTER replay selector than surprise-at-harvest
  or freshness?** (d) tests retro vs the monitor's freshness proxy, but the
  cleanest scientific question is a three-way selector comparison (retrodiction-
  decay vs harvest-surprise vs 1/age) at matched replay volume. That may deserve
  its own clause or its own registration — flagged for the lead as a possible
  scope expansion, not silently included.
- **Q4 — H measured in chunks vs tokens.** The ladder is specified in chunks
  {2,8,32,128}; the carrier decay laws (P35) are in tokens (τ≈2000, wall≈16k).
  K=64 makes H=128 chunks ≈ 8k tokens, comfortably inside the band, but if the
  store's bindings span multiple chunks the effective lag is fuzzier. OPEN
  whether to define H in tokens-since-write for precision. Low-risk (both put
  the ladder inside the measured band) but should be fixed before the run.
- **Q5 — does the foreign-state control (C2) need seed-matching or
  recipe-matching only?** A foreign organism at seed+1 shares architecture but
  not init; a stronger control is the SAME organism at a DIFFERENT stream
  position (its own future state, which also never wrote these specific
  bindings). OPEN which is the tighter specificity control; possibly run both.

---

## 7. One-paragraph summary for the register header (draft)

*MS-E / retrodiction organ, keyed v1: P41 v0 read null because a bulk-histogram
probe on mean-pooled features sees the state's marginal, not chunk-specific
memory. v1 replaces the learned marginal-reader with the organism's OWN keyed
holographic read — value bound to key by phase, read by de-rotation — and
measures its decay over a backward H-ladder {2,8,32,128} as a live per-chunk
retention meter (the F3 knee read backward, on the horizon P35 already walled).
Two controls separate binding from marginal: a shuffled-key contrast (kills
"reads a prior") and a foreign-organism read (kills "reads a generic decay
law"). The organ then triggers consolidation on MEASURED per-binding decay and
replays exactly the decayed bindings — the precision version of P54's dividend
monitor — and is scored against that monitor at equal budget, with a
shuffled-trigger control deciding whether WHICH bindings were replayed carries
the recovery gain. F1 symmetry made literal: forward surprise says learn,
backward surprise says consolidate.*
