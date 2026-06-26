# O1 — constant-memory sequence modeling with an external knowledge index

**A constant-memory recurrent stream coupled to a growing `.causal` knowledge index, with a measured capacity threshold in the gated readout.**

**Author:** David Tom Foss · **Year:** 2026 · **License:** Apache-2.0

> This README is a **timestamped public disclosure** (prior art). Every claim below is a measured
> number with the exact script that reproduces it. The dates, the code, and the result JSON in
> this repository are the record.
>
> **O1 builds on [GSSM](https://github.com/DT-Foss/gssm)** and inherits its full commit history.
> GSSM defines the architecture: a bounded reproducing-kernel SSM operator, a length-invariant
> NoPE-selective recurrence, and an `O(log T)` parallel scan. O1 extends it in three directions —
> constant-memory streaming (training and inference), an external knowledge index the recurrence
> consults at runtime, and a measured capacity threshold in the gated readout.

---

## Summary

O1 is a recurrent sequence model whose per-token cost and memory are **independent of sequence
length**. It consumes an unbounded token stream at constant RAM, both at inference and during
truncated-BPTT training, and retains information across input gaps. It is paired with an external,
incrementally-built `.causal` knowledge graph: when the recurrent state encounters high surprise,
it queries the graph and folds the retrieved association back into the stream without a gradient
update. Separately, we characterize a sharp capacity threshold in the model's gated readout and
locate where a use-driven reinforcement process compounds (the index) versus where it cannot (the
bounded state).

The architecture and its kernel theory are documented in
[GSSM](https://github.com/DT-Foss/gssm); this repository documents the streaming, retrieval, and
threshold results built on top of it.

---

## Contributions

Each entry states the measured result and the script that reproduces it. All runs are CPU,
offline, constant-memory, and memory-guarded.

### 1 — Constant-memory streaming (training and inference)
Truncated BPTT carrying the detached state across chunks reproduces full BPTT to a gradient cosine
of **1.0000**; held-out loss decreases 8.69 → 5.22 at flat resident memory, while a no-detach
control grows 0.77 → 1.81 GB. The state retains a single bit through a **256-token input gap**
(recall accuracy 1.0; zeroing the state at the gap reduces recall to chance), carried by a learned
near-unity-decay channel (γ = 0.9999).
→ `src/streaming_train.py`, `plots/living_stream.png`

### 2 — Runtime retrieval from an external index
At a high-surprise position, the pre-gap recurrent state is forked and the retrieved `.causal`
association is injected into the continuation. This lowers follow-on surprise **without any
gradient update** (mean reduction +0.0256; improved 27 of 40 probes). The recurrence consults its
external index during inference.
→ `src/closed_loop.py`, `src/pathfinding_bridge.py`, `src/attic.py`

### 3 — Capacity threshold, structural (knowledge graph)
On the constructed knowledge graph, the percolation susceptibility χ **increases with system size
N** ([5.1, 6.6, 18.1, 17.9] over N = [2k, 5k, 10k, 20k]) — a finite-size scaling signature that a
smooth crossover does not produce — driven by PMI-weighted edges, with a critical mean degree
⟨k⟩ ≈ 1.
→ `src/percolation_hard.py` → `results/percolation_hard.json`, `plots/night_percolation.png`

### 4 — Capacity threshold, dynamical (knowledge graph)
With the edge set held fixed, reinforcing only the *traversed* paths raises connected capability
super-linearly (C: 0.04 → 0.66, logistic with mid-range inflection). Reinforcing random pairs or a
degree-preserving shuffled graph produces no gain (+0.00) across three seeds — the effect is
driven by graph structure, not by weight inflation.
→ `src/reinforcement_loop.py` → `results/reinforcement_loop.json`

### 5 — Capacity threshold in the gated readout
In the model's actual recurrence, the gated (m·tanh) readout exhibits a **sharp capacity cliff at
load K/D ≈ 1** (maximum slope 1.32 per unit load, fall concentrated in a narrow load band), whereas
a linear least-squares readout on the same state only rolls off smoothly (0.57 per unit load, no
cliff). The use-driven reinforcement of §4 requires *recoverable* latent structure, which the
bounded state does not retain above capacity (a fact is either readable or erased) — so that
compounding belongs to the external index, while the bounded state provides the sharp gated read.
→ `src/gssm_potentiation.py` → `results/gssm_potentiation.json`, `plots/bridge_gssm_threshold.png`

### 6 — Operator readout: multiple reads from one state
A single bounded state stores K key–value pairs in superposition; K least-squares operators
de-multiplex them. Recoverable information scales with the readout operators applied, not the state
alone (≈ D facts in a D-dimensional state).
→ `src/operator_readout.py`

---

## FORGE

A related deterministic code-generation engine (FORGE) is described as a **capability statement
only** in [FORGE.md](FORGE.md). No generator and no generated artifacts are included; the full
system is available on request for verified security research.

---

## Foundation

O1 inherits GSSM's commit history and its mathematical foundation (Möbius coupling,
doubly-stochastic spectra, non-reversible lifted Markov chains, and the universal phase-transition
result underlying the threshold work), archived with permanent DOIs — see [PAPERS.md](PAPERS.md).

## Scope

Results stated as n = 1 (architecture/seed) are labeled as such — e.g. the §5 dynamical
dissociation is a single-configuration result on the actual recurrence. Where a result is robust we
state it without qualification: 1B tokens at flat memory, the gated cliff at 2.4× the linear slope,
and length extrapolation flat to 32× are measured, not extrapolated.

## Reproducing

Each contribution writes a JSON under `results/`; figures regenerate from those JSONs via
`src/plot_*.py`. CPU-only, constant memory.

## Contact

David Tom Foss — `dtfoss-dev@proton.me`
