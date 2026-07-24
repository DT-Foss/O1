# F4 — The two-system law

**Statement (from `FOUNDATIONS.md` F4).** Bounded states read through saturating gates exhibit a
sharp capacity cliff — above capacity, stored structure is *deleted*, not gracefully degraded.
Therefore any system that must accumulate unbounded knowledge over an unbounded stream divides by
construction into (i) a bounded state carrying few live bindings with a sharp read, and (ii) an
external, growable index written and consulted under the surprise calculus (F1). The state does
not compound; the index does. This is not a design choice — it follows from a measured property
of the gate.

The arc below is four separate measurements that chain into one law: a cliff, a theorem for why
the cliff is where it is, a hybrid measurement of what the index buys back, and a closed-loop
measurement of the same effect on a real language stream.

---

## 1. The cliff — gated reads fail sharply, linear reads fail gracefully

`src/gssm_potentiation.py`, `results/gssm_potentiation.json`. Same bounded state (`D=64`), two
readouts, swept across load `K/D`:

| load K/D | gated fidelity | linear fidelity |
|---|---|---|
| 0.25 | 0.9999 | 1.0 |
| 0.75 | 0.9813 | 1.0 |
| **1.00** | **0.6519** | 0.9905 |
| 1.25 | 0.3324 | 0.8493 |
| 3.00 | 0.1129 | 0.3642 |

The gated readout's cliff: `cliff_at_load = 1.0`, `cliff_drop = 0.3294`, `sharp_cliff = true`,
**max slope 1.318** per unit load, over a transition span of only 0.5. The linear readout on the
*identical underlying state*: `cliff_at_load = 1.25`, `sharp_cliff = false`, **max slope 0.565**,
transition span 1.25 — more than double the width, less than half the slope. Same substrate, same
information — the failure mode is a property of the *readout*, not the state. This is the
structural reason the two-system split is forced: a gated read (the kind any practical
architecture uses at inference) does not fail gracefully near capacity, so anything that must
keep accumulating cannot live inside it.

## 2. The Rank-1 theorem — *why* the cliff sits at load ≈ 1

`analysis/RANK1_CAPACITY_THEOREM.md`. A bank of `D` scalar leaky-integrator channels is, per
channel, a **rank-1 functional** of its feature history (`z_t = ⟨w_t, Φ_t⟩`, PROVEN). Associative
recall over `K` keys is a rank-`K` (outer-product) requirement; a `D`-channel scalar bank with a
fixed linear readout can recover at most `D` bindings exactly (Eckart–Young ceiling). But the
*mechanism* matters more than the channel count: a leaky scalar channel performs **key-agnostic
accumulation** — no operation in the recurrence conditions the write on a match between a stored
key and an incoming query — so the achievable **binding rank per channel is 1**, and the trained
stack's effective binding rank `D_eff` is `O(1)`, far below the raw channel count `D=128`.
Measured: on `K=8, V=64` MQAR (`paper/evidence_companion/hybrid_B.json` — the paper's
Task-B harness, `src/stack_mqar_run.py`; note this is a DIFFERENT instrument from
`src/mqar.py`, whose scalar floor sits at ~0.017 — two instruments, two floors, both real),
pure-Selective recall **0.1406/0.1445**
(train/test) lands almost exactly on the closed-form `D_eff=1` prediction **0.139**; inverting the
bound gives `D_eff ≈ 1.02`. Attention (rank `K`) and the Selective+attention hybrid (SSAS) both
hit **1.000** — the double dissociation that discharges the theorem's one modeling assumption.
This is the same cliff from §1, now with a closed-form location: the bounded state's binding
capacity is not "large but finite," it is **rank-bounded at essentially 1 per channel**, which is
why the cliff sits at load ≈ 1 and not at load ≈ D.

## 3. The hybrid measurement — what the index buys back above the cliff

`src/holo_index_hybrid.py`, `results/holo_hybrid.json` (P16 checkpoint, `gate_rate=1.0`, seeds
0/1, `chance=0.0625`):

| P | base (state alone) | hybrid (state + index) | Δ |
|---|---|---|---|
| 16, G=8, seed0 | 0.16 | 0.485 | +0.325 |
| 16, G=8, seed1 | 0.155 | 0.415 | +0.26 |
| 16, G=32, seed0 | 0.20 | 0.615 | +0.415 |
| 16, G=32, seed1 | 0.10 | 0.41 | +0.31 |

Across the four P=16 cells at full consultation rate, base ranges **0.10–0.20**, hybrid ranges
**0.41–0.615** — a consistent 2–4× lift, dose-dependent on `gate_rate` (the `gate0.5` cells sit
between base and the `gate1.0` cells in every row of the raw sweep) and guarded against
random injection (`random` sits at or below `base` in every cell — a wrong reminder does not
help). Runtime index consultation, with **no gradient update to the base model**, lifts recall
far past what the bounded state alone can hold. (Full acceptance gate on this sweep — `hybrid@P16
≥ 0.9` — was **not met**; the honest ceiling of an *untrained* consultation read is diagnosed
below, not hidden.)

Training the model **with** stochastic consultation in-stream changes the picture again —
`src/holo_reminded.py`, `results/holo_reminded.json`:

| P | G | base | hybrid (reminded) |
|---|---|---|---|
| 2 | 8 | 0.5475 | **0.715** |
| 2 | 32 | 0.5375 | **0.8425** |
| 2 (seed2) | 8 | 0.535 | **1.00** |
| 2 (seed2) | 32 | 0.615 | **1.00** |
| 8 | 8 | 0.20 | 0.5925 |
| 16 | 8/32 | 0.1575–0.205 | 0.45–0.53 |

At `P=2` a model trained to expect reminders reads them at **0.99–1.00** (individual seed cells
hit 1.0 exactly); the *untrained*-consultation ceiling (§ above) tops out lower even with perfect
injection, because the base model never learned to *use* an injected reminder as such. The lesson
the reminded sweep adds: the index is not free just by existing — the stream has to be trained to
consult it, and once it is, near-perfect recall is recoverable even where the bounded state alone
is at 0.20 (`P=8`, five times chance but nowhere near ceiling).

**Held-out-key control** (`results/holo_heldout_keys.json`, P=2): both the phase-carrying arm and
the magnitude-only baseline generalize to keys never seen as keys during training — accuracy on
`test_keys` (e.g. `holo_carried|test_keys|G32`: **0.525**) tracks `train_keys` (**0.565**) within
noise, and the zeroed-at-gap null collapses to chance (`holo_zeroed_at_gap|train_keys|G0`:
**0.06**, `test_keys|G0`: **0.095**, both `beats_chance_3x: false`). This kills the "it's just a
lookup table over memorized keys" explanation for either mechanism — the binding generalizes.

## 4. The consequence, closed-loop, on a real stream — the state consults, the index compounds

`src/closed_loop.py`, `results/closed_loop.json`. On a live language stream (`wikitext.causal`),
40 detected knowledge gaps: injecting the retrieved `.causal` path back into the stream as tokens
through the same `O(1)` state, **with no gradient update**, drops follow-on surprise by
**mean 0.0256 nats**, helping **27 of 40** gaps (67.5%). This is the F4 law observed end-to-end,
outside the synthetic MQAR harness: the bounded state does not need to have memorized the fact —
it needs only to consult an index that has it, fold the retrieval back in as ordinary tokens
(the F2 layout-decoupling license is what makes "fork state, inject tokens, continue" a coherent
operation), and move on. The state provides the sharp, cheap read; the index provides the
capacity; F1's surprise signal is what triggers the hand-off between them.

---

## 5. P34, scored — the rank hypothesis dies on mqar, the SNR fallback engages

The Rank-1 theorem (§2) leaves one conjecture open: does the *complex/phase* state (F3) lift
per-channel binding rank above the scalar ceiling `D_eff ≈ 1`? P34 pre-registered a relational
test on the reproducible `src/mqar.py` instrument (the original 0.1406 Task-B anchor's generating
script is lost — a documented reproduction debt, not a new problem) with an explicit fallback: if
`D_eff` stays flat, the phase's capacity rent is not rank, and the honest alternative is SNR-based.

**Scored 2026-07-24, full grid** (`results/rank_sweep_final.json`, K∈{2,4,8,16,32} × 4 arms × 4
seeds): the harness itself is calibrated — attention validity min recall **0.9898**, scalar anchor
reproduction **exact** (0.0166 vs anchor 0.0170±0.0022). The rank criterion then **fails both
sub-clauses as registered**: at the one eligible cell (K=2, 3/4 seeds ignited) the phase/scalar
`D_eff` ratio is **0.371** — phase capacity per channel *lower* than scalar, the opposite sign of
the prediction — and no higher K clears 3× chance on seed-mean to even be scorable. The cliff-ratio
clause is flat at **1.0** (no shift). **The registered fallback therefore engages as written: the
phase's capacity rent is not per-channel rank — SNR is now the standing hypothesis** for why phase
memory pays a cost under load. The dominating phenomenon underneath the failed rank test is itself
a finding: phase **ignition collapses with load** (3/4 → 1/4 → 1/4 → 0/4 → 0/4 ignited seeds across
rising K) — whatever capacity the phase channel could offer at high K, training reliability to
reach it dies first. This is a clean negative, named as one: rank-lift was the wrong mechanism,
stated plainly rather than left as an open conjecture, and it sharpens rather than weakens the
two-system law — the index (§3, §4) is not competing with a rank advantage the state secretly has;
the state's ceiling is real and mechanism-identified, which is exactly why the external index is
not optional scaffolding but the required second system.

---

## What this chains into

Cliff (§1, measured slope 1.318 vs 0.57) → theorem for the cliff's location (§2, `D_eff ≈ 1.02`
measured) → hybrid measurement of what an external index buys back above the cliff (§3, 0.15→0.51
class of lift, reminded reads at 0.99–1.00) → the same effect, closed-loop, on real text (§4,
+0.0256 nats, 27/40) → the one candidate escape from the cliff (phase-lifted rank) tested and
closed, fallback mechanism now standing (§5). Five independent instruments, same law: **compounding
belongs to the index; the bounded state consults it, and it cannot rank its way out of that.**

## What is not yet measured

- The full `hybrid@P16 ≥ 0.9` acceptance target from the pre-registered P16 prediction was not
  met by the untrained-consultation sweep (ceiling ~0.615 at best, not 0.9); the reminded sweep
  (§3, in-training consultation) is the designed next lever and partially closes the gap at low
  `P`, but P=16 reminded (0.45–0.53) has not been pushed to the same near-1.0 regime P=2 reaches.
  `[not yet measured — candidate experiment: reminded training at P=16 with a longer curriculum or
  larger gate_rate schedule]`
- P34's SNR-based fallback (§5) is now the standing hypothesis but is itself untested — no SNR
  model of the phase channel's capacity rent has been fit or scored against the collapsing-ignition
  data. `[not yet measured — candidate experiment: fit an SNR-vs-load model to the ignition-rate
  curve (3/4→1/4→1/4→0/4→0/4) and test its predicted capacity independently of the rank framing]`
