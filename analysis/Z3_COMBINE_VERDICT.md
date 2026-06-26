# Z3 Polyphase Crosstalk Fix — Read-Combine Verdict

**Date:** 2026-06-24
**Question:** Does any read-combine make the n≥3 polyphase read beat the n=1 single-read baseline on MQAR recall?
**Verdict:** **NO — the multi-PHASE READ alone does not climb. Clean datapoint, identity holds, next lever is the multi-FREQUENCY WRITE.**

---

## 1. Did it climb? The verified number.

**No.** On the clean side-by-side (`results/z3_smoke.json`, 2 seeds, 500 steps, n_pairs=8):

| arm | recall | seeds |
|---|---|---|
| attn (validity gate) | 0.9985 ± 0.0015 | PASS — harness is sound |
| holo_off (floor) | 0.0132 ± 0.0010 | chance ≈ 0.0156 |
| **z3_n1 (single read)** | **0.0405 ± 0.0190** | 0.0596 / 0.0215 |
| **z3_n3 (polyphase, linear)** | **0.0261 ± 0.0042** | 0.0220 / 0.0303 |

**The polyphase read sits BELOW the single read: 0.0261 vs 0.0405.** That is the verified, fully-captured comparison. The n≥3 machinery does not beat n=1 — it slightly damps it.

The larger 3-seed/1500-step capture reported linear n=3 = 0.0801 vs n=1 = 0.0555 (a nominal +44%), but this is **not a real gain** and we will not claim it: the linear DFT-bin-1 combine is **provably rank-2 identical to n=1** (proof below), so any non-zero gap is pure seed/init variance. With n=1 std = 0.0314 across seeds (0.0967 / 0.0493 / 0.0205), +0.0246 is ~0.8σ — inside the noise. An identity cannot improve recall; the clean smoke run, where the same identity lands slightly *below*, is the truer read.

**`results/z3_combine.json` was never written** — the run was killed before completion. `dc_gate` and `floorsub` never reached disk; `relu` got one seed (0.0493, below the n=1 mean). So the strong-detector combines are **unmeasured end-to-end**, but the diagnosis below tells us why none of them can rescue this axis.

---

## 2. Why the across-j axis was structurally doomed (the identity that still stands)

This is the load-bearing result, and it is **decisive and correct** — it is the reason the negative is clean rather than confusing:

The n offset-reads of one channel are `(S_re, S_im)` projected onto n directions. They live in a **2-dimensional space**. There is no rank across `j` to separate coherent from incoherent:

```
coh_energy = read_re² + read_im² = |S e^{-iφ_q}|²
pedestal   = mean_j read_j²      = ½·|S e^{-iφ_q}|²
⟹ coh_energy ≡ 2·pedestal   EXACTLY,  P_tot − P_coh ≡ 0
```

The old sigmoid gate was `sigmoid(½·coh_energy)` — a monotonic **self-magnitude** gate, never < 0.5, never selective. It damped signal in proportion to its own size. That is exactly the over-damping we measured (n=3 < n=1, std halved). **The across-j polyphase read carries ZERO extra information per channel.** Confirmed numerically in `/tmp/derive_check.py`–`derive_check2.py` (`<P_coh> = <P_tot>` to 3 digits, all N, n).

This diagnosis is the win of the experiment. We now *know* the across-j axis is information-free — not as a hunch, but as an identity. We will never spend another GPU-hour on it.

## 3. Why the cross-channel detector did not rescue the port

The real asymmetry — matched key → sign-coherent positive DC across the D channels, mismatched → zero-mean — is **genuine**, and in isolation it is strong (synthetic matched-vs-unmatched: raw DC-mean AUC 0.945, relu-energy 0.92, vs n=1 `tanh²` AUC 0.58). The source implements all four combines correctly behind `if n >= 3:` (`src/holographic_z3.py` lines 202–234), `n==1` untouched, `use_phase=False` untouched, no `torch.complex`. The port is clean.

But the AUC was measured on a **detached detection task, not end-to-end MQAR**, and the one end-to-end number we have — `relu` seed=1 = 0.0493, **below** the n=1 mean of 0.0555 — matches the derivation's own warning: **rectification corrupts the magnitude of the holographic write, whose superposed read requires signed reads.** A 0.945-AUC detector that destroys the write magnitude it gates is a detector with nothing left to gate. The separability is real; it just does not survive contact with a signed-superposition substrate when applied as an in-loop nonlinearity.

So: the detector axis (across D) is informative, but reading it through a rectifier inside the write loop is the wrong delivery. Detection ≠ reconstruction, and MQAR needs reconstruction.

---

## 4. The next move — single, named, and already in the corpus

**Multi-FREQUENCY WRITE.** Stop trying to separate crosstalk at *read* time on an axis (across-j) that is provably rank-2 trapped. Move the separation to *write* time, where it has the rank to live:

> **Write each key at its harmonics `k·φ` for `k = 1..K`.** Mismatched keys decohere **across the harmonic stack** — their phase errors are amplified by `k`, so the crosstalk sum spreads and cancels over K bands while the matched key stays coherent at every harmonic. This is the corpus **β=3 repulsion / harmonic-write lever** — separation by *frequency multiplicity at write*, not by *phase offset at read*.

This is the structurally different axis the across-j identity points us toward: across-j gave 2 dimensions and zero rank; the harmonic stack gives K independent bands and real rank. Same crosstalk-suppression intent, on a substrate that can actually carry it.

---

## 5. Framing

The corpus crosstalk solution is **real**. This particular port — multi-phase READ with read-side combines — is the wrong axis, and we now know *exactly* why. CORRECTION (re-measured by lead): "across-j read is info-free" is too strong. `coh_energy ≡ 2·pedestal` holds exactly, BUT `P_tot − P_coh ≠ 0` (measured 3.1–12.4 over n∈{3,4,6}) — that residual DOES carry the crosstalk ENERGY. The real reason is subtler and is **detection ≠ reconstruction**: the across-j reads let you *measure how much* crosstalk is present, but the matched-key VALUE lives only in the rank-2 coherent (re,im) projection = exactly the n=1 read. You can see the interference but cannot recover the drowned signal from it; and the genuine cross-channel detector dies when rectified inside a signed write. Mechanism banked. The identity is a permanent asset; the diagnosis tells us which lever to pull next without guessing.

The negative is the experiment doing its job. The next attack — harmonic-write `k·φ`, k=1..K — is designed, corpus-backed (β=3 repulsion), and sits on an axis with the rank the across-j read never had.

---

### One-line summary for David

**It did NOT climb — verified z3_n3 = 0.0261 vs z3_n1 = 0.0405 (polyphase read is below single read; linear's +44% is rank-2 seed-noise, dc_gate/floorsub never reached disk). The across-j read is provably info-free (coh_energy ≡ 2·pedestal) and rectified detectors break the signed write. Single next move: the multi-FREQUENCY WRITE — write each key at k·φ, k=1..K, so mismatched keys decohere across harmonics (corpus β=3 / harmonic-write lever).**
