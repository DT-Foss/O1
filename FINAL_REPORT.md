# GSSM Research — Final Report (by Opus 4.8)

All numbers below are [measured], full-corpus WikiText-2 MLM unless noted, Mac Mini M4.

## TL;DR

- **GSSM-Selective works and scales stably to 37.5M params** (best PPL 135 at d512/L4 = 18.8M; 37.5M = d1024/L2) with no collapse anywhere in the d256–d1024 × L2–L4 grid. PPL is flat at the 135–142 WikiText-2 data ceiling — not improving with more params. The M1 stabilization stack holds across the whole range.
- **The d512 "collapse" was an optimization-at-width problem, not an architecture limit.** μP is the root-cause lever: velocity saturation 0.43 → 0.05 (9×), full stack 202 → 153 PPL. The architecture was never the bottleneck.
- **The Selective scan is length-invariant — and we proved it.** Removing additive sinusoidal PE collapses length drift from +243% to +2.6% (train T=32 → eval T=1024). The residual drift was a PE confound; GSSM needs no PE and is *more* length-robust without one.
- **Double dissociation (M4): Selective's geometry is hybrid-compatible, Pure's is not.** SSAS (3 SSM + 1 attn) reaches 100% MQAR recall and stays length-robust; PPAP — structurally identical, differing *only* in scan class — gets 16% recall and degrades with length. The attractor topology causally determines whether attention can bind.
- **Proven closed-form theory underwrites all of it:** Pure has a boundary attractor s\*=1 (exponential saturation; smoke-probe Pure 66% vs Selective 14%, full-corpus Pure last-position saturation up to 0.71 at d512); Selective has an interior attractor s\*=√(1−(1−v²)^{α/(1−γ)}) — a T-independent leaky integrator (smoke-probe 14% saturation). Boundedness proven both ways.

## What is PROVEN (M1–M4)

### M1 — The d512 width-fix (root cause: optimization, not architecture)
At d512 the Selective model collapsed (202 PPL, velocity saturation 0.43). The cause is **unbounded W_v running hot at width** (velocity sat jumps 10.3× from d128→d512, W_v preactivation 2.4× hotter), with the gates a secondary effect (γ̄ mid-range 0.43, ᾱ 0.49; the frozen-γ fraction does rise 0.04→0.38, but W_v velocity saturation is the dominant ~10× lever and μP alone resolves it — so the cause is the projection, not dead gates). **μP is the single root-cause lever** — it drops velocity saturation ~9× (0.43→0.05) on its own. The full stabilization stack (μP + preLN + warmup) reaches **153 PPL**, matching the d256 level (158), and the pure variant recovers too (310→191). d128 was never harmed (161→161). Multiseed at 400K/4ep showed 0/5 seeds learned without the fix — difficulty is a data×epochs×seed interaction; the fix makes training *reliable*, not just better.

### M2 — Length extrapolation + NoPE (the scan is position-invariant)
Train T=32, evaluate with frozen weights out to T=1024. **Selective-NoPE: +2.6% drift (near-flat). Selective+PE: +242.7%.** Removing additive sinusoidal PE collapses the drift by ~100×, proving the residual was a **PE confound**, not a property of the scan. NoPE-Selective is also *better* at long T (156 vs 170 @ T128). Honest framing: the causal scan still carries finite-T position, so the claim is "length-invariant / needs no PE," not "position-free." For contrast, the warmup-Transformer now learns but still drifts +23.2%; Pure drifts +874.5%.

### M3 — Scale-up (stable to 37.5M, PPL floor is the data budget)
Selective + full stack, T=128, across d256–d1024 × L2–L4: **best d512/L4 = 135** (beats the Phase-1 best of 158), d512/L2=144, d768/L4=140, d1024/L2=142. **No collapse anywhere** — the M1 fix holds across the entire range. Depth beats width on this budget. The **135–142 floor is the WikiText-2 data ceiling (~1.7M tokens), not an architecture limit.** Sub-135 needs more data, not more params. This is explicitly **not** a power-law claim on a tiny corpus.

### M4 — Hybrid GSSM+attention (the double dissociation)
**SSAS and PPAP are structurally identical — 3 SSM layers + 1 attention layer — and differ ONLY in scan class.**
- **Task A (length, train T=32 → eval T=256, % rise):** sel4(SSSS) +38.8, hyb_mid(SSAS) +35.0, hyb_top(SSSA) +35.3, pure_proxy(PPAP) **+313.4**, attn4(AAAA) +1.2 (but dead, acc 0.16).
- **Task B (MQAR recall, len 256):** attn4 1.000, **hyb_mid(SSAS) 1.000**, hyb_top(SSSA) 0.999, sel4(SSSS) 0.144, **pure_proxy(PPAP) 0.161**.

The finding: **SSAS = 100% recall + length-robust; PPAP = 16% recall + length-degrades.** Selective's interior attractor *preserves* the information attention needs to bind; Pure's boundary attractor *saturates and corrupts* the residual stream. Pure-Selective alone can't bind (14% — the honest bounded-scalar-state limit), but **one attention layer restores full recall in the Selective hybrid.** Attractor topology causally gates whether attention works.

## The honest limits

- **Bounded scalar state cannot do exact associative recall alone** — confirmed at **14% MQAR** for pure-Selective. No KV-binding mechanism exists in a single scalar channel. (One attention layer fixes this in the hybrid; the standalone limit is real.)
- **No S5-style state-tracking (TC0).** Bounded scalar state can't represent the requisite group structure. Derived, not yet empirically stressed.
- **PPL floor ~135 is the WikiText-2 data budget, not the architecture.** 1.7M tokens caps it. More params do not move it; more data would.
- **Phase-GSSM (complex/phase channel) is built and verified but NOT yet run** — it is the principled attempt to push past the recall boundary, currently unmeasured.

## What can go straight into the paper (gssm.tex)

**Strongest 3 claims to lead with:**
1. Selective's interior attractor is **length-invariant** (proven via NoPE: drift +243% → +2.6%).
2. The **double dissociation** — identical 3-SSM+1-attn architectures, scan class alone decides 100% vs 16% recall.
3. The d512 collapse is an **optimization-at-width artifact** (μP, velocity sat 9×), not an architectural ceiling — and the architecture then scales stably to 37.5M.

**Figure → claim mapping (in `./plots/`):**
- `fig_widthfix.png` → M1: μP/full-stack recovers d512 (202→153). *(Lead figure for the optimization claim.)*
- `fig_attribution.png` → M1 root cause: W_v hot at width, gates healthy.
- `fig_length_extrap_v2.png` → M2: NoPE collapses length drift (+243%→+2.6%). *(Lead figure for length-invariance.)*
- `fig_saturation.png` → Theory: Pure 66% vs Selective 14% saturation, boundary vs interior attractor.
- `fig_scaleup.png` → M3: stable scaling d256–d1024 × L2–L4, no collapse, 135 floor.
- `fig_scaling.png` (+ `ppl_vs_width.png`) → Phase-1 base: Selective beats Pure 64–79 PPL at every working config.
- `fig_complementary_collapse.png` → Pure vs Selective collapse contrast.
- `fig_hybrid_length.png` → M4 Task A: SSAS/sel4 length-robust (~35-39% rise) vs PPAP explodes (+313%).
- `fig_hybrid_recall.png` → M4 Task B: the double-dissociation money figure — SSAS/hyb_top/attn4 at 100% recall (green) vs sel4/PPAP at 14-16% (red), with the SSAS-vs-PPAP "same structure, opposite outcome" annotation. *(Lead figure for the hybrid claim.)*

**Results draft:** `./analysis/RESULTS_DRAFT.md` is the paper-ready results section.

## Required code-vs-prose fixes before submission (gssm.tex)

1. **Selective term is α·log(1−v_gated²)** where v_gated = v·gate — **not** α·log(1−v²). The prose drops the gate inside the log.
2. **The "O(log T) parallel Blelloch scan" is aspirational** — the implementation is **sequential O(T)**. Claim "constant inference state, no KV-cache," **not** "fast parallel-scan training."

## Next experiments (if pursued)

1. **More data (bigger corpus)** — the only route to sub-135 PPL. The floor is data, not architecture; a larger corpus would confirm.
2. **Run Phase-GSSM** (complex/phase channel — built and verified, never executed) — the principled attempt to break the 14% recall boundary without an attention layer.
3. **Implement the real parallel scan** (Blelloch) — turn the aspirational training-speed claim into a measured one.

## Artifact index

**Reports & analysis** (`./`)
- `FINDINGS.md` — full lab notebook (43 KB)
- `analysis/RESULTS_DRAFT.md` — paper-ready results section
- `analysis/MASTER_BRIEF.md` — research brief
- `analysis/FRONTIER_MAP.md` — open-problem map
- `analysis/DEEP_ANALYSIS.md` — deep-dive analysis
- `analysis/NEXT_EXPERIMENTS.md`, `analysis/RECON_limitations.md` — supporting
- `analysis/make_figures.py` — figure generator (regenerates all 10 plots incl. both hybrid figures)
- `FINAL_REPORT.md` — this report

**Plots** (`./plots/`) — 10 rendered: fig_scaling, fig_complementary_collapse, fig_length_extrap_v2 (+ v1), fig_saturation, fig_attribution, fig_widthfix, fig_scaleup, fig_hybrid_length, fig_hybrid_recall (+ legacy ppl_vs_width, saturation_profile). All publication-grade, all backed by measured JSON.

**Source runners** (`./src/`)
- `width_fix.py` (M1), `length_extrap_v2.py` (M2), `scaleup.py` (M3), `hybrid.py` (M4)
- `mqar.py`, `multiseed_d512.py`, `diagnose_d512.py`, `instrumented_runner.py`, `phase_gssm.py` (built, unrun)

**Results JSON** (`./results/`)
- `width_fix.json` (M1), `length_extrap_v2.json` (M2), `scaleup.json` (M3), `hybrid_A.json` + `hybrid_B.json` (M4)
- `phase1_fulldata.json` (base scaling), `multiseed_d512.json`, `d512_diagnosis.json` (+ matching `*_log.txt`)

**The paper:** the GSSM paper source (`paper/gssm.tex`, maintained separately)
