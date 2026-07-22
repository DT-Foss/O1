# The Holographic Carrier — why the phase survives the silence

*2026-07-22, written between the launch of the WP4 gap sweep and its results;
the predictions derived here are registered in PREDICTIONS.md (P8–P11).*

This note unifies the repo's two proven memory mechanisms — the **γ-carrier**
(idle-persistence, streaming_train.py Pillar E: a write-once-freeze channel
with γ≈0.9999, α shut on fillers) and the **key-conditioned holographic
write** (holographic_gssm.py: S_t = γ_t·S_{t-1} + a_t·e^{iφ_t}, read by
de-rotation) — into one claim about the streaming-gap regime, with
closed-form predictions.

## 1. The gap acts on magnitude only — the phase is written, never evolved

The complex accumulator's recurrence multiplies by a **real** γ_t ∈ (0,1):

    S_t = γ_t · S_{t-1} + a_t · e^{iφ_t}

During an input gap in which the write drive vanishes (a_t → 0 — exactly the
α-shut-on-filler behavior the beacon task *measurably grows*, Pillar E), the
recurrence degenerates to S_t = γ_t·S_{t-1}: a pure real scaling. Therefore:

**The phase of every channel is invariant across the gap.** arg(S) — the key
binding — does not rotate, precess, or diffuse; there is no imaginary part in
the decay path to move it. The only thing a gap can do is shrink |S| by
Γ(G) = ∏_{gap} γ_t ≈ exp(−(1−γ)·G).

Two immediate consequences:

- **Persistence is a γ problem, and γ is learnable.** Recall survives the gap
  iff |S| stays above the readout's noise floor: the knee sits at
  G* ≈ ln(|S₀|/m_min)/(1−γ). The measured trained carrier (γ = 0.9999,
  τ ≈ 10³) puts G* in the thousands of tokens. Training on a gap task grows
  exactly this channel — that is the *same* mechanism idle-persistence
  already proved for one bit, now carrying a keyed binding.
- **Predicted shape: a plateau, not a decay.** Once a γ→1 carrier forms,
  recall vs G is flat until G*, then falls off a cliff. Prediction P8:
  accuracy(G=128) within 10 pp of accuracy(G=8) for ignited cells. An
  exponential-looking recall-vs-G curve would mean no carrier formed
  (curriculum failure), not that the mechanism is wrong — the two are
  distinguishable by the γ-spectrum of the trained model.

## 2. The pair count acts on phase space only — the 1/√P law

With P bindings superposed in one channel, the de-rotated read at query q is

    Re(S·e^{−iφ_q}) = m_q + Σ_{p≠q} m_p·cos(φ_p − φ_q)

For quasi-random key angles the crosstalk term has zero mean and std
~ m·√((P−1)/2): SNR ∝ 1/√(P−1) — the classic HRR/VSA holographic law the
repo already measured on fixed-length MQAR (25.8% @ 2 pairs → 7.6% @ 8,
crosstalk_smoking_gun). Nothing about the gap enters this expression.

## 3. The synthesis: two orthogonal axes, and what that buys

The streaming-gap experiment separates the memory problem into two
independent coordinates:

| axis | attacks | defended by | learnable? |
|---|---|---|---|
| gap length G | magnitude \|S\| | γ→1 + α-shut (write-once-freeze) | **yes — proven** (Pillar E) |
| pair count P | phase SNR | more rank (slots/heads/index) | structurally capped ~1/√P per channel |

Hence the factorization prediction **P10**: recall(P,G) ≈ f(P)·g(G), with
g → plateau after training. The gap does not touch the phase; the pairs do
not touch the decay.

**The disruptive reading, if the factorization holds:** a bounded O(1)
complex state does *perfect* keyed recall in the regime of few concurrently
open bindings held across arbitrary silence (P small, G unbounded) — which is
the regime of a *life*: at any moment a stream holds a handful of open
questions, carried across long stretches where they are not mentioned.
Attention's KV cache is not buying persistence (the state gets that for free,
at O(1)); it is buying *capacity* — many simultaneous bindings. And capacity
is exactly the axis O1 already assigns to the external index (contributions
6–8: the state/index split derived from the gated-readout cliff). The same
two-system architecture falls out of an independent derivation: **state =
few bindings across unbounded time; index = unbounded bindings at retrieval
cost.** That is the thesis of the whole repo, reached from the phase algebra.

## 4. Falsifiers

1. Phase drift during gaps (arg(S) measurably rotating while a≈0) — would
   contradict §1 directly; measurable from the φ-carrier probe internals.
2. Recall-vs-G exponential *despite* a formed γ≈1 carrier — kills the
   plateau claim (P8) even where ignition succeeded.
3. No factorization (G-profiles differ strongly across P) — kills §3's
   independence claim (P10).
4. holo_off == holo_on at P≥2 with both well above chance — the phase pays
   no rent in streaming; the selective magnitude state suffices (P11).

Every falsifier is computable from results/holo_stream_recall.json plus one
internals probe — no new training required.
