# FOUNDATIONS — the primitives underlying the o1-state architecture

**Author: David Tom Foss · Public technical disclosure, first published in this
repository 2026-07-24 (commit timestamp is authoritative). This document
deliberately discloses each foundation in its broadest form, including
contemplated variants and generalizations beyond the specific implementations
in `src/`. Everything below is prior art as of its commit date.**

The measured results in this repository (constant-memory billion-token
streaming, 4096× length extrapolation, surprise-gated training at ~25%
gradient tokens, keyed holographic recall across silence with a movable decay
knee, dosed sleep consolidation, runtime index consultation) are
*demonstrations*. This document states the six primitives that produce them,
each formulated over the **class** of systems it applies to — not over the
specific networks used to measure it.

---

## F1 — The surprise calculus: self-measured prediction error as the universal control signal

**Statement.** In any streaming learner that carries a persistent state and
emits predictions, the learner's *own instantaneous prediction error* (any
monotone functional of it: per-token NLL, chunk means, rolling quantiles,
z-scores, ratios against a reference model, ensembles thereof) suffices as
the control signal for every plasticity and memory decision the system makes:

- **when to learn** — gradient updates gated on the error signal clearing a
  data-dependent threshold (rolling quantile, fixed bar, adaptive
  homeostat), applied at any granularity (token, span, chunk, batch, layer,
  parameter group);
- **what to remember** — spans, keys, or abstractions selected for storage
  in any external memory when the signal spikes;
- **when to consult** — retrieval from any external store triggered by the
  signal, with the retrieved content folded back into the stream (as tokens,
  as state perturbations, or as auxiliary inputs) with or without a gradient;
- **when and how much to sleep** — offline replay of stored material dosed
  by measured marginal benefit (a dividend monitor), throttled to zero when
  the benefit is spent;
- **how curious to be** — the gating threshold itself regulated by the
  statistics of the signal (homeostatic target rates, non-stationarity
  detectors), including the selection *between* input sources by per-source
  signal statistics.

This applies to any bounded-state sequence model (linear SSMs and gated
variants, RNNs, hybrid attention/SSM stacks), any training regime (from
scratch, continued, fine-tuned), and any stream (text, code, sensor data,
multimodal token streams). Measured instantiations: `src/pos_run.py` (gating,
~25% gradient tokens at ≈0.97–1.00 of full-gradient learning),
`src/pos_index.py` (storage + recurrence probes), `src/closed_loop.py`
(consultation), `src/pos_sleep.py` / `src/pos_sleep_cycles.py` (dosed
replay; the dividend life-curve), `src/active_sourcing.py` (source
selection).

**Contemplated variants disclosed here:** per-layer and per-head gating;
surprise signals computed against an exponential-moving-average teacher copy;
gating of optimizer moments separately from gradients; threshold schedules
tied to wall-clock duty cycles ("waking hours"); multi-signal calculi
combining surprise with uncertainty (entropy) and disagreement (ensemble
variance).

## F2 — The exactness license: bounded contraction makes streaming training exact and decouples training layout from deployment layout

**Statement.** For any recurrent operator whose state dynamics are a
contraction with an effectively bounded receptive field (measurably: the
gradient of the output at time t with respect to inputs at t−k decays below
numerical relevance for k beyond a small horizon r), the following are exact,
not approximate:

1. **Detach-carry streaming training**: truncated BPTT with the state carried
   and detached at chunk boundaries reproduces full-window BPTT gradients
   (measured: cosine 1.0, max-abs-delta 0.0, `results/streaming_check.json`)
   whenever the chunk exceeds r.
2. **Layout decoupling**: full-sequence forward with zero initial state and
   chunked-carried forward are the same operator (measured to float
   precision for the selective scalar scan, the complex holographic scan,
   and biased-γ variants). Therefore the *training* computation graph
   (full-sequence, arbitrarily long, gradient reaching every write) and the
   *deployment* computation (chunked, O(chunk) memory, unbounded length) may
   be chosen independently — train however the gradient needs, deploy
   however memory requires. This license is what turns gap-curriculum
   training into deployable streaming skills (`src/holo_gap_knee.py`,
   `src/holo_mag_read.py`).

Applies to any member of the affine-scan family (`src/ssm_family_reduction.py`
reduces Mamba/S6, S5, LRU to one operator at ~1e-15) and to any stacked
combination with pointwise/feedforward layers.

## F3 — Phase–magnitude separation in bound complex states: content is written, persistence decays, and the two never mix

**Statement.** In any bounded recurrence that stores associations as
complex-valued (or otherwise rotational) accumulations `S_t = γ_t·S_{t-1} +
a_t·e^{iφ(x_t)}` with real decay γ, the stored *content* (the phase — the
key binding) is invariant during input silence, while only the *magnitude*
(the persistence) decays as ∏γ. Consequences, each measured or in
measurement:

- recall over silence exhibits a **knee, not a slope** — flat until the
  magnitude crosses the readout floor, at G* ≈ ln(margin)/(1−γ);
- the knee is **movable by any lever that raises effective γ on
  non-informative inputs** (learned input-gating, curriculum, explicit
  γ-bias initialization of a subset of channels) — measured arc: knee 32→512
  in one day (`analysis/HOLO_STREAM_VERDICT.md`);
- the readout floor is **independently movable by magnitude-invariant
  reads** (normalizing |S| before de-rotation, or reading pure phase),
  because the content is intact by construction (`src/holo_mag_read.py`);
- capacity (how many bindings) and persistence (how long) are **independent
  axes**: pair count attacks phase SNR (~1/√P), gap length attacks only
  magnitude.

Disclosed variants: multi-slot and multi-head phase banks; per-channel
γ-kickstart at any bias point; phase-only readouts; renormalization applied
in-state at controlled intervals (a "magnitude refresh") rather than at
read; binding angles derived from learned key projections, from token
embeddings directly, or from external key registries.

## F4 — The two-system law: sharp gated readouts force compounding into an external index

**Statement.** Bounded states read through saturating gates exhibit a sharp
capacity cliff (measured: fidelity 0.99→0.65 across load K/D≈1, slope 1.32
vs 0.57 for linear reads, `src/gssm_potentiation.py`) — above capacity,
stored structure is deleted, not gracefully degraded. Therefore any system
that must *accumulate* unbounded knowledge over an unbounded stream divides
by construction into: (i) a bounded state carrying few live bindings with a
sharp read, and (ii) an external, growable index (symbolic graph, span
store, key-value registry — any persistence layer) written and consulted
under the surprise calculus (F1). Runtime consultation without any gradient
measurably lifts performance far beyond state capacity (recall 0.15→0.51 at
P=16, dose-dependent, `src/holo_index_hybrid.py`), and models trained *with*
in-stream consultation read reminders at ~1.0 fidelity
(`src/holo_reminded.py`). Disclosed variants: shared indices across multiple
independent organisms (collective memory); freshness-weighted and
dividend-monitored replay from the index; index entries as reminders
injected in-stream at any position; consultation policies trained end-to-end.

## F5 — Operating modes are family-generic: the calculus attaches to the operator class, not to one architecture

**Statement.** Because the linear-SSM family reduces to a single affine scan
operator (machine-precision reductions in `src/ssm_family_reduction.py`),
every operating mode above (F1's gating/storage/consultation/sleep, F2's
streaming exactness, F3's phase memory where the state is complex) is
defined on the *family*, not on any single parametrization. A
surprise-gated Mamba, a sleeping S5, an LRU with a bolted-on phase bank and
an external index are instances of the same disclosed system.
(`src/pos_family_transfer.py` measures the transfer directly.)

## F6 — Train short, deploy unbounded: shift-equivariance plus the exactness license remove every length wall

**Statement.** An operator with no absolute-position term (NoPE; the only
index-dependence is through lags) is in-distribution at every sequence
length; combined with F2, training at tiny horizons (T=32, gap≤12) yields
deployment at unbounded horizons (measured: PPL ×0.98 at 4096× training
length; 1B streamed tokens at flat 4.36 GB; keyed recall flat across 8
detached chunk boundaries). Length, in this architecture class, is a
wall-clock quantity, never a memory or validity quantity.

---

## Interactions (disclosed as a system)

The six foundations compose into a single organism — one process that
streams unboundedly (F6) at exact constant memory (F2), decides every
plasticity and memory action from its own surprise (F1), carries live keyed
bindings through silence in phase (F3), accumulates unbounded knowledge in
an external index it writes and consults in flight (F4), and does all of
this identically across the SSM family (F5). A reference composition is
specified as CHIMERA in `analysis/MOONSHOTS.md` (MS6); partial compositions
are already measured throughout `results/`.

*Every claim above is either measured in this repository (file references
inline) or explicitly disclosed here as a contemplated variant. Measured
numbers carry their reproduction scripts; the registered-prediction ledger
(`analysis/PREDICTIONS.md`, scored by `src/score_predictions.py`) documents
which quantitative expectations survived contact with the data — including
the ones that did not.*
