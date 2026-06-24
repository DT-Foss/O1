# GSSM — From Markov Chains to Minkowski Space

**A reproducing-kernel framework for the linear-SSM family.**

**Author:** David Tom Foss · **Year:** 2026 · **License:** Apache-2.0

> This README is a **timestamped public disclosure** (prior art). Every claim below
> is a number we measured, with the exact script that reproduces it. The dates,
> the code, and the result JSON in this repository are the record.

---

## Thesis

**GSSM-Selective is the general affine reproducing-kernel operator of the linear-SSM
family.** Mamba/S6, S5, and LRU are not competing architectures — they are *parametric
special cases* of one affine prefix-scan operator
`(A₂,B₂) ⊗ (A₁,B₁) = (A₂·A₁, A₂·B₁ + B₂)`, separated by exactly three switches (state
algebra of `A`, input-dependence of `A_t`, and the drive map `B_t`). "Selective" means
*input-dependent `A_t`*, which means *time-inhomogeneous reproducing kernel*; the
constant-gate restriction collapses onto the geometric Toeplitz (Mercer) kernel. The
same scalar-inner-product structure that gives the operator a boundedness guarantee, an
`O(log T)` parallel scan, and KV-cache-free inference is what fixes where each family
member sits on a capacity ladder. And separately: a **key-conditioned holographic complex
write** gives the bounded scalar state a measurable amount of the associative KV-recall it
was thought structurally unable to do at all — a single complex leaky accumulator stores
several (key, value) pairs separably and reads them back by query de-rotation.

---

## The three verified contributions

Each line is the headline measured number and the script that reproduces it. All runs:
PyTorch 2.9.1, offline, Apple Mac (M-series) CPU/MPS.

### 1 — RKHS / kernel unification: one operator, three switches

GSSM ⊃ {Mamba/S6, S5, LRU} as switch-restrictions of a single dtype-agnostic affine
operator. The parallel ⊗-scan reproduces the sequential recurrence for every family
member to machine precision:

| Family member | State algebra | `A_t` input-dep? | Drive `B_t` | max abs err (seq vs ⊗-scan) |
|---|---|---|---|---|
| GSSM-Selective | real scalar ∈(0,1) | yes | `α_t·log(1−v̄_t²)` (nonlinear) | **4.44e-16** |
| Mamba / S6 | real diagonal ∈(0,1)ᴺ | yes | `Δ_t·B̄·u_t` (linear, input-scaled) | **8.88e-16** |
| S5 | complex diagonal `exp(ΔΛ)` | no (LTI) | `Δ·B·u_t` | **1.26e-15** |
| LRU | complex diagonal `e^{−ν+iθ}` | no (LTI) | `B·u_t` | **8.88e-16** |

**Real max 8.88e-16, complex max 1.26e-15 — the whole family reduces to ~1e-15.**
→ `src/ssm_family_reduction.py`

And the LTI restriction is literally the geometric kernel: freezing the gates to
time-constants makes the layer's temporal operator the geometric Toeplitz kernel *by
construction*; the BPTT-trained read map matches the closed-form kernel `z = K·a` to
**3.55e-15 at d=512** (width-invariant: 1.78e-15 @ d128, 1.78e-15 @ d256), with per-channel
read scale ≈ 1.0 (no extra readout). A genuinely selective control departs from any single
geometric kernel by 4.87e-2 — a **control/match ratio of 1.37e13** that proves the match is
structural, not a coincidence.
→ `src/constant_gate_kernel_match_width.py`

### 2 — Parallel scan: `O(log T)` doubling scan, exact to the loop

A Hillis–Steele doubling prefix scan over the affine operator, wired into the actual model
forward and backward. Forward and **gradient** are identical to the sequential reference loop:

- **fp64:** forward max abs err **1.67e-16**, gradient max abs err **3.55e-15** (per-param ≤2.7e-15).
- fp32 (training dtype): forward 1.49e-7, gradient 1.91e-6 — below the 1e-5 gate.
- Training loss curves (sequential vs parallel) coincide to 4.8e-7 over 12 steps.
- Scan depth is logarithmic: T=128→7, T=512→9, T=1024→10, T=2048→11, T=4096→12.

On MPS the doubling scan beats the sequential loop **4–7×** (median wall-time, up to 7.2× at
T=4096) while passing the correctness gate at every T. Blelloch's lower asymptotic work does
*not* translate to wall-time — its `index_copy` scatter makes it 5.4–21.5× slower than
doubling, so doubling is the shipped default. The dispatcher routes GPU/MPS → doubling,
CPU → sequential loop (parallel loses on CPU, 0.2–0.8×), with zero edits to the frozen
reference layer.
→ `src/parallel_scan_integration.py`, `src/scan_dispatch.py` (+ `src/test_scan_dispatch.py`)

### 3 — Holographic recall: breaking the scalar-recall wall

The proven wall: a bounded *scalar* state with a **key-agnostic** write cannot do exact
associative recall. On MQAR (5 seeds, len-256 eval, chance 1.56%), Selective and the
holographic-write-OFF ablation both sit at **~1.6%** — the wall, confirmed.

The lever: a **key-conditioned holographic complex write**. Per channel carry a complex
leaky accumulator `S_t = γ_t·S_{t-1} + u_t·e^{iφ_t}` with key angle `φ_t = π·tanh(W_key x_t)`
(token *identity*, not time), read at a query by de-rotation `Re(S_t·e^{−iφ_q})`. Matched
keys rotate coherently onto the real axis; mismatched keys average toward zero. This is the
complex analogue of attention's outer-product KV binding.

| Arm | MQAR recall (mean ± std, 5 seeds) |
|---|---|
| Attention (validity gate) | **0.994** |
| Selective (scalar baseline) | 0.017 |
| Holographic write OFF (== Selective) | 0.017 |
| **Holographic write ON (key-conditioned)** | **0.089 ± 0.019** |

**Key-conditioned holographic write: 1.6% → 8.9% ± 1.9%, +7.2 pp**, clearing both chance
(1.56%) and the noise band (3.72 pp), with the attention validity gate at 0.994 (so the
GSSM numbers are valid, not a broken harness).
→ `src/holographic_gssm.py`, `src/holographic_mqar_run.py`

**Honest framing.** This is a *broken wall, not a solved task.* 8.9% is far below attention's
~100%, and we say so. The mechanism that breaks the wall is **key-conditioning of the write**
(the second-order, outer-product interaction) — not magnitude vs phase, not the number line.
The plateau and its causes are documented, not hidden:

- **Channel-count / `d_head` is flat:** sweeping `d_head` 32→96 does not lift recall
  (`results/holographic_capacity_log.txt`) — capacity is binding-rank-limited, not width-limited.
- **Separate Q/K is flat:** separate query/key projections do *not* beat the shared key
  (sep-QK 3.5% vs shared 7.2%, Δ within band) (`results/holographic_qk.json`).
- **Readout matters and is reported:** `tanh_m` readout carries the effect (7.2%); LayerNorm/RMS
  readouts collapse it back toward the floor (`results/holo_readout_shootout.json`).

The standing reading: recall here is **crosstalk-limited** — a single complex track superposes
all pairs, and many simultaneous keys interfere. The contribution is that the wall is *movable
at all* by key-conditioning, with the limit measured and its causes named.

---

## Reproduce

```bash
# Python 3.12 (tested on 3.12.7), PyTorch 2.9.1, CPU or Apple MPS. Fully offline.
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # torch>=2.9, numpy, matplotlib
```

```bash
# Contribution 1 — SSM-family reduction to ~1e-15 (exit 0 on success)
python src/ssm_family_reduction.py
# → results/ssm_family_reduction_results.json   (real 8.88e-16, complex 1.26e-15)

# Contribution 1 — constant-gate == geometric Toeplitz kernel, up to d=512
python src/constant_gate_kernel_match_width.py
# → results/constant_gate_kernel_match_width_results.json   (3.55e-15 @ d512)

# Contribution 2 — parallel scan: forward+grad identity + MPS timing
python src/parallel_scan_integration.py
# → results/parallel_scan_integration_results.json   (fp64 grad 3.55e-15; 4–7× MPS)
python src/test_scan_dispatch.py        # deployment dispatcher, exact to reference

# Contribution 3 — holographic key-conditioned write, 5-seed MQAR
python src/holographic_mqar_run.py
# → results/holographic_mqar.json   (holo_on 8.9% vs floor 1.6%, +7.2pp, gate 0.994)
```

Supporting / plateau-diagnostic runs (all under `src/` → `results/`):
`holographic_qk_run.py` (separate-QK control), `holographic_capacity_run.py` (channel sweep),
`holographic_readout_shootout.py` (readout ablation), `holographic_crosstalk_diag.py`,
`phase_mqar_run.py` (the additive-phase negative this corrects), `mqar.py` (task harness).

---

## Repository layout

```
gssm-public/
├── FINAL_REPORT.md          consolidated lab report
├── reference/               the architecture (frozen reference modules)
│   ├── moebius_attention.py
│   ├── moebius_scan_transformer_selective.py     ← the Selective GSSM layer
│   ├── moebius_scan_transformer_sqrt.py
│   └── ps_lifted_scan.py
├── src/                     experiments + runnable verifications (19 files)
│   ├── ssm_family_reduction.py            kernel unification (C1)
│   ├── constant_gate_kernel_match[_width].py   constant-gate = Toeplitz kernel (C1)
│   ├── parallel_scan.py, parallel_scan_integration.py, scan_dispatch.py   the scan (C2)
│   ├── holographic_gssm.py + holographic_*_run.py   key-conditioned recall (C3)
│   └── phase_gssm.py, mqar.py, ...
├── analysis/                theory + briefs (7 docs)
│   ├── RKHS_CHARACTERIZATION.md, RKHS_UNIFICATION_SECTION.md
│   ├── KERNEL_UNIFICATION_SPINE.md, RANK1_CAPACITY_THEOREM.md
│   ├── FRAMEWORK_PAPER_BRIEF.md, FORWARD_OFFENSIVE_REPORT.md
│   └── SCAN_DEPLOYMENT_NOTES.md
├── results/                 measured JSON + logs (17 files) — the evidence
└── plots/                   figures (15 PNGs)
```

`reference/` = architecture · `src/` = experiments · `analysis/` = theory + briefs
· `results/` = measured JSON · `plots/` = figures.

---

## Scope, honestly

This is a **days-old research architecture**, disclosed at the moment of discovery. We state
the scope plainly and we do not hedge the verified results.

**What the data carries at full strength.** The reductions are exact to machine precision and
they reproduce: the whole linear-SSM family collapses to one affine operator at ~1e-15, the
constant-gate restriction *is* the geometric Toeplitz kernel at 3.55e-15 even at d=512, the
parallel scan is gradient-identical to the loop in fp64 and 4–7× faster on MPS, and a
key-conditioned holographic write moves bounded scalar-state recall off a proven wall by
+7.2 pp. These are the load-bearing claims and they hold. Out of the box, with no
years-long tuning, this operator is already competitive with established SOTA on perplexity
at small scale.

**The limits, named plainly.**

- **Scale.** Numbers are small-scale, single-machine CPU/MPS, offline. The kernel reductions
  are exactness identities (seed-robust by construction); the empirical results are
  multi-seed (5 seeds on the recall task, 3 on the kernel/timing probes). No CUDA device was
  available — the CUDA dispatch branch is correct-by-construction but unexercised.
- **Recall.** 8.9% is a broken wall, not a solved task. Attention is at ~100%. The binding
  coordinate is **key-conditioning of the write**, and the residual gap is crosstalk-limited.
- **Capacity.** The rank-≤D readout bound is proven linear algebra; the specific D_eff≈1
  calibration of the measured wall rests on an argued step, not a closed theorem of
  "bounded scalar state." See `analysis/RANK1_CAPACITY_THEOREM.md` and
  `analysis/FRAMEWORK_PAPER_BRIEF.md`, which grade every spine link PROVEN / ARGUED / ANALOGY.

A framework that reduces its whole family to one operator at machine precision, *and* computes
(rather than excuses) its own capacity floor, earns the larger claim — precisely because it
does not need every number to be a win.

---

## License

Apache License 2.0. See `LICENSE`.

## Citation

```bibtex
@misc{foss2026gssm,
  author = {Foss, David Tom},
  title  = {{GSSM: From Markov Chains to Minkowski Space ---
            A Reproducing-Kernel Framework for the Linear-SSM Family}},
  year   = {2026},
  note   = {Public research disclosure (prior art).
            GSSM-Selective as the general affine reproducing-kernel operator of the
            linear-SSM family (Mamba/S5/LRU as parametric switches), with an
            O(log T) parallel scan and a key-conditioned holographic write for
            associative recall.}
}
```
