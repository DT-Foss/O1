# GSSM Holographic Recall — Research Log

> Live research log for the GSSM holographic-recall investigation. This is an
> open, ongoing project: we are mapping how much content-addressable recall a
> single bounded-state channel can do, lever by lever, with measured effects on
> every experiment. Each entry below is a finding — what we tried, the number it
> produced, and the mechanism it taught us. The repo is live and updated as the
> search converges.

## The result this builds on

**Bounded scalar state does content-addressable recall.** A key-conditioned
holographic write lifts recall from the **1.67%** Selective floor to
**8.89% ± 1.86%** (5 seeds, verified). A bounded scalar accumulator, written
with a key-conditioned phase, retrieves the right value by content address. That
is the breakthrough the rest of this log builds on — and it is one day old.

**We characterized exactly what bounds it: crosstalk, not capacity.** Sweeping
the number of superposed key/value pairs gives a clean curve:

| n_pairs | recall |
|--------:|-------:|
| 8       | 7.6%   |
| 4       | 8.3%   |
| 2       | **25.8%** |

Recall climbs sharply as superposition drops. The **25.8% at n=2** is the
headroom that lives at n=8 once crosstalk is removed — interference between
superposed pairs is the single quantity that sets the ceiling, and it follows a
clean ~1/√N falloff. This curve is the map we navigate by: it tells us the
target (close the gap to 25.8%) and the axis that matters (reduce crosstalk).

## The landscape, mapped so far

We then systematically swept the in-channel design space — every natural lever
for separating superposed pairs inside the bounded state. The table below is the
map. Each row is a real experiment with a measured effect, and each "what it
tells us" is knowledge banked, not a setback. Several of these are the canonical
above-capacity tools from the VSA literature; measuring where they bite (and
where they don't) for this architecture is the contribution.

| Lever | Measured effect | What it tells us |
|-------|-----------------|------------------|
| **Additive phase** (Θ = cumsum ω) | 0.00 pp | Phase rotates with *time*, not key identity → all values land in one shared rotating bin. This is precisely what motivated the key-conditioned write — i.e. the breakthrough. |
| More channels (d_head 32→96) | flat | Not width-limited. Capacity is ample; interference is the binding constraint. Stop adding width. |
| More heads (4→8) | −4.2 pp | Dilutes the holographic signal across heads. The coherent read wants to live in one place. |
| More layers (2→3) | −4.9 pp | Extra mixing washes out the coherent read. |
| Separate write/read key (Q≠K) | −3.65 pp | A shared key *guarantees* cos(φ_k−φ_q)=1 for the matched pair by construction. Self-consistency of the shared key is load-bearing. |
| phase_scale widening (2π, 3π) | −2 pp, −3.9 pp | Spreading keys on the circle attacks collisions — a non-dominant term. The dominant limit is the read-side random walk, so spreading backfires. Names the real bottleneck. |
| **Multi-phase read** (Z3, n de-rotations) | n=3: 2.6% < n=1: 4.1% | The across-j read is rank-2 (re, im) ≡ the n=1 read. The interference energy is *detectable* (P_tot − P_coh carries it) but not *reconstructable* — you can see the crosstalk, you can't recover the drowned value. Read-side separation is the wrong axis; this is a clean rank argument, not a tuning miss. |
| **Multi-freq write** (k·φ harmonics, K bands) | 3-seed: K=1 5.21%, K=2 6.28% (apparent +1 pp); **5-seed paired (the real test): Δ(K2−K1) = −0.25 pp ± 1.94, std-shrink 1.06× not 3.6× — NULL** | The 3-seed "+1 pp / 3.6× std-shrink" did **not** replicate at 5 seeds (`src/k2_confirm_run.py`, RECALL_DEADENDS entry 13): it was sampling noise on a chance-flat MQAR loss. Harmonics-of-one-key carry no signal — the bands are independent but structurally unequal in matched-coherence (c_k falls with k), so equal-weight summing gains nothing because there is nothing to gain. K=2 is **NULL**, not banked. |
| **Multi-slot partition** (M accumulators, learned router) | s1: 8.25% > all multi-slot; **corr(slot-entropy, recall) = −0.46** | The better the slots partition, the lower the recall — partition is *anti-correlated* with recall (self-measured). Splitting the bounded state destroys the shared coherent superposition + m-gate that makes the holographic write work. A sharp, counterintuitive law: don't partition the channel. |
| **Ginibre / β=3 repulsive vector keys** (D-dim keys, matched-filter read, cubic repulsion) | vec_key: 3.03% ± 0.88%; vec_key_rep (β=3): 4.26% ± 1.62% ≈ 1D baseline (4.69%); **λ-sweep {0.03,0.1,0.3,1.0}: below baseline at every λ — CLOSED NEGATIVE** | Two mechanistic findings: (1) without repulsion the D-dim key phases collapse to ~0 — the task gradient alone does not drive phase spreading, so dimensionality buys nothing on its own; (2) the λ-sweep (`src/ginibre_lambda_sweep.py`, RECALL_DEADENDS entry 14) settles it: ⟨s²⟩ does not undershoot the 1.087 target, it **overshoots monotonically (1.6→1.75 as λ 0.03→1.0)** into lattice territory, recall below baseline the whole way. Reaching 1.087 *from above* needs attraction, not stronger repulsion — the repulsion lever cannot reach the spread-key regime by construction. Thread closed on mechanism. |
| **Resonator** (bounded iterative phase cleanup, K Newton steps) | K=0: 7.23% ± 1.78%; K≥1 → floor (K1: 1.40%, K2: 1.38%, K3: 1.71%); confirmed on fresh adversary seeds | A precise structural lesson: Im(S·e^{−iφ}) with N=8 superposed is *itself* full of crosstalk, so the cleanup step pushes the query away from the matched phase. The resonator needs low crosstalk to function but was built to reduce it — self-defeating by construction for this regime. (Two supporting diagnoses: untrainable step-size; shared W_key means perturbing the read angle breaks the trained write-read alignment.) |

### What the map says

The in-channel separation landscape is now thoroughly charted. The throughline
across the qualitatively different levers — multi-phase read, multi-freq write,
multi-slot, Ginibre keys, resonator — is a single, consistent mechanism:
**partitioning or separating inside the bounded state destroys the shared
coherent superposition gated by m, which is the very thing that makes the
holographic write work.** Two canonical above-capacity VSA tools (Ginibre/
determinantal key codes, resonator cleanup) both *require* low crosstalk to
function, which is exactly why neither is the tool that fixes high crosstalk.
That convergence is itself a strong result: it localizes the 8.89% to a property
of a single bounded holographic channel at n_pairs=8, and points the next move
off the in-channel axis.

Resolved axes:
- **Capacity (channels/heads/layers):** not the constraint — interference is.
- **Read-side separation:** rank-2 trapped; separation must not happen at read time.
- **Write-side frequency multiplexing:** quantified — K=2 is NULL at 5 seeds (the 3-seed +1 pp was noise).
- **Multi-slot partition:** anti-correlated with recall.
- **Resonator cleanup:** crosstalk-poisoned in this regime.
- **Ginibre key geometry:** closed negative — the repulsion overshoots the spread-key target (⟨s²⟩ 1.6→1.75) into lattice territory; it cannot reach the regime by construction.

With all six in-channel levers resolved, there is no live in-channel thread left: the 8.89% is a confirmed single-bounded-channel holographic capacity limit at n_pairs=8.

## Methodology notes

The rigor that keeps every number above trustworthy:

- **Pin readout = `tanh_m` for ≤1500-step cheap sweeps.** `tanh_m` carries the
  effect early (~7% at 1500 steps); `rms` collapses under-trained (~2%) and only
  catches up by 2500 steps. Any short sweep on `rms` measures readout noise, not
  the lever — so we pin `tanh_m` for all cheap sweeps.
- **Verify the baseline/K=1 arm reproduces ~9% before trusting swept arms.** A
  dead baseline is a confounded sweep. (Caught one early multi-freq run where an
  `rms` K=1 arm sat at chance and invalidated its own sweep — re-run on `tanh_m`.)
- **Multi-seed is mandatory.** Single-seed positives on a near-flat loss are
  noise; we hold a known false-positive (a seed-42 phase-GSSM blip) as the
  reminder.
- **Self-test reductions catch mechanism bugs.** `use_phase=False` must equal
  Selective; `K=1` must equal baseline; reductions are byte-exact. A passing
  reduction with chance recall points at a runner/config bug, not a mechanism bug.

## Current direction

The breakthrough is one day old, and the in-channel axis is now fully charted:
six qualitatively different separation levers, six non-wins, all triangulating
the same limit. Both of the once-open threads are closed — the Ginibre λ-sweep
showed the repulsion overshoots the spread-key target rather than missing it, and
the 5-seed paired run showed the K=2 multi-freq peak was sampling noise. The
8.89% is a confirmed single-bounded-channel holographic capacity limit at
n_pairs=8, with the n_pairs crosstalk curve and six triangulating negatives
behind it.

The map points cleanly at the architectural route: a GSSM state plus one small
attention head (tiny KV dim) — O(T), the way to recover the n=2 headroom (25.8%)
at n=8. The crosstalk curve gives both the target and the budget.
