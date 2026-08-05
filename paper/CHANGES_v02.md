# CHANGES — paper v0.1 → v0.2

**v0.1:** Draft v0.1, July 24, 2026 — 8 pages, 13 references, 4 figures.
**v0.2:** Draft v0.2, August 5, 2026 — 14 pages, 13 references, 7 figures, 3 tables.

Every number below was copied out of the result JSON named beside it, not from
the v0.1 text. Where the two disagreed, the artifact won — v0.1 predates
several corrections.

---

## 1. What entered v0.2 (new sections)

| § | Section | Artifact |
|---|---|---|
| 5 | **The width curve: a two-point law, falsified by its own third point** | `results/gate_law_width_curve.json`, `results/gate_rate_width_probe.json` |
| 6 | **The exactness license, swept** | `results/f2_equivalence_sweep.json` |
| 7 | **Seven billion tokens in one life** | `results/lifetime_7b_curve.json`, `results/lifetime_7b_series.json` |
| 8 | **Length, re-anchored** | `results/scale_to_a_million.json`, `results/length_extrap_v2_extreme.json` |
| 10 | **Two live machines, one organism** | `results/moebius_parity.json` |
| 11 | **The replay-cost law: two falsifications and the mechanism behind them** | `results/p39_two_machine_scored.json`, `results/cadence_probe.json` |

New figures (all rendered from results JSONs by `src/plot_v02_figures.py`,
house style inherited from `paper/figures/make_o1state_figures.py`):

- `fig_gate_width_curve.pdf` — the falsified two-point trend and the
  per-gradient-token inversion that survives it
- `fig_f2_sweep.pdf` — gradient exactness vs. overlap; layout decoupling at 0.0
- `fig_length_reanchor.pdf` — ×0.803 at 524,288×, plus the PE/gate control
- `fig_lifetime_7b.pdf` — already rendered by `src/plot_lifetime_7b.py`, now placed

The falsifications ledger (§12) gained a visible itemized subsection naming
four killed predictions with the numbers that killed them.

---

## 2. Claims that CHANGED STRENGTH

### 2.1 The gate law and width — **weakened as registered, sharpened in substance**

- **v0.1 / the P42 ledger entry:** "the gate law GROWS with width" — ratio
  0.9729 (d=128) → 0.9953 (d=256).
- **v0.2:** that reading is **falsified by the third point**. The ratio runs
  0.9729 → 0.9953 → 0.9777: a maximum at d=256, not a monotone rise.
- **What replaced it:** the arms are not equally dosed (d=512 gates at 19.84%
  vs 24.72%), and **per gradient token spent d=512 is the best of the three**
  (0.3547 vs 0.2969 / 0.2937). Selection got cheaper with width, not worse.
- Both candidate explanations for the smaller dose are reported as **refuted**
  from `gate_rate_width_probe.json` (distribution shape flat across width;
  window drift does not separate d=256 from d=512 — both at 0.230 above q75),
  with the rate located inside ignition (0.2042 vs 0.1785 at the first eval)
  and the open question stated as needing an instrumented ignition run.

### 2.2 F2, the exactness license — **strengthened in scope, bounded in condition**

- **v0.1:** "grad-cosine 1.0000, max-abs-delta 0.0" from one operator at one
  (chunk, overlap) point (`streaming_check.json`), stated unconditionally
  across "twelve independent equivalence measurements ... all pass at 0.0–1e-6".
- **v0.2:** swept over **4 operators × 6 (chunk, overlap) points**. Two separate
  statements, only one unconditional:
  - layout decoupling **exact to 0.0** at every operator and chunk size down to 16;
  - gradient exactness **gated by warmup overlap, not chunk length** — cosine
    1.000000000000 / rel err ~5e-7 at overlap 16; rel err 5e-3–2e-1 and worst
    cosine **0.9762** at overlap 0.
- The v0.1 blanket "all pass at 0.0–1e-6" would be **wrong** against this sweep
  and was removed. F2 now ships with its measured failure mode attached.

### 2.3 Length — **re-anchored upward**

- **v0.1:** "flat perplexity at 4096× the training length", "billion streamed
  tokens at 4.36 GB flat RSS", "×0.98 at 4096×".
- **v0.2:** the strongest committed length artifact is `scale_to_a_million.json`:
  **×0.803 at 524,288×** the training length, RSS 1.90 → 2.48 GB, PPL improving
  **monotonically** (1.000 / 0.896 / 0.886 / 0.825 / 0.803). Flatness understated
  the result; the curve improves.
- Control added from `length_extrap_v2_extreme.json`: NoPE ×0.973 vs
  Selective+PE ×4.23 vs Pure ×11.25 at 256× — **the wall is the position term,
  not the length**.

### 2.4 Portability — **from checkpoint-resume to two live machines**

- **v0.1:** cross-ISA migration stated as checkpoint→resume, heldout
  6.182391 == 6.182391 (retained in v0.2 under F7).
- **v0.2:** adds the live staging — two machines, real network, real `scp`/`ssh`,
  **source never paused**: at the one equal chunk count (560) heldout delta
  **0.0 exact** (ARM 6.199778 == x86 6.199778), reproduced across two
  independent stagings.
- **Limit stated as the artifact states it:** parity is defined *only* at equal
  chunk counts. B on 16 cores overtakes A (B at 5480 chunks, A max 3000), so
  multi-cycle parity needs rate-matched machines.

### 2.5 Stop-free migration — **falsified as registered** (new negative)

- **v0.1:** "stop-free migration ... achieves bit-identical parity at every
  scale tried" (F7 paragraph). That sentence covered check (b) only, but read
  as if the whole prediction passed.
- **v0.2:** (b) still passes everywhere. **(a) and (c) are falsified as
  registered on two independent machines**: (a) 7.085 / 5.603 vs threshold
  < 3.0; (c) 2.219 / 2.039 vs threshold < 1.0.
- Mechanism located and reported: **replay costs ~2× live streaming**
  (45.4 s vs 22.3 s CPU for identical chunks), both ISAs, contended or not.
  The shared-core explanation is **refuted** (16 free cores move (c) by 0.18).
- Cadence lesson added as a method finding: toy chunk weight inflated the
  earlier ratios (17.935 / 3.789 → 7.085 / 2.219 at production cadence), because
  snapshot size scales with the model, not chunk weight (7.94 → 2.59 chunk slots).

### 2.6 Scale limitation — **widened**

- **v0.1:** "Scale: d_model ≤ 128".
- **v0.2:** "d_model ≤ 512" — the width curve runs to d=512. The limitations
  section also gained: the lifetime constancy claim is about resources not
  learning; parity only at equal chunk counts; replay ~2× live; gradient
  exactness needs overlap; phase ignition fragile in *load* as well as seed and
  reduction order.

---

## 3. Corrections vs v0.1 (artifact wins)

| v0.1 said | Artifact says | Where |
|---|---|---|
| gate law grows with width (2 points) | maximum at d=256, two-point trend falsified | `gate_law_width_curve.json` |
| twelve equivalence measurements "all pass at 0.0–1e-6" | worst grad cosine 0.9762, worst rel err 2.2e-1 at overlap 0 | `f2_equivalence_sweep.json` |
| flat PPL at 4096×, ×0.98 | ×0.803 at 524,288×, improving monotonically | `scale_to_a_million.json` |
| stop-free migration parity "at every scale tried" | (b) yes; (a)/(c) falsified as registered on two machines | `p39_two_machine_scored.json` |
| "P1–P39 at this writing" | P1–P42 | `analysis/PREDICTIONS.md` |
| d_model ≤ 128 | d_model ≤ 512 | `gate_law_width_curve.json` |
| twin gate rates "identical to four decimals" | 0.2521154 vs 0.2521634 — equal to **three** decimals; v0.2 prints both | `pos_summary.json` |
| shuffled A-spans "as useless as no replay" | shuffled 0.317260 is **better** than no replay 0.375721; the registered check is only that shuffled does not beat private (0.233120) | `pos_shared_index.json` |
| beacon carrier "γ ≈ 0.9995" | the beacon-swap carrier measures **γ = 0.9998** (α 0.0133 in gap vs 0.5652 at beacon); 0.9995 belongs to the P35 gap-ladder carrier, a different run | `beacon_swap.json` |

These three were found by re-deriving each inherited v0.1 claim from its JSON
rather than trusting the v0.1 text. All other inherited numbers reproduced
exactly: ratio 1.0090681, 8.658815 → 4.742984 / 4.778174, grad-token fraction
0.2516660, twin excess 0.0029320 and twin final 4.736495 (`pos_summary.json`);
injection −0.0865997 vs −0.2792804, helped 0.348 vs 0.102 (same file); surgery
equivalence 6.67572e-06 and growth-over-fresh +0.1269 / +0.1655 nats
(`hot_swap_growth*.json`); family transfer 0.9804 / 0.9552 / 0.9677 over three
seeds with the head-to-head lead 0.156 / 0.148 / 0.133 nats (`pos_family*.json`);
shared-index forgetting 0.155783 / 0.233120 = 0.668 (`pos_shared_index.json`);
kill-rejoin delta −0.014127 and sleep-over-idle +0.064565
(`portable_organism_core.json`); beacon surgery and swap recall 1.000 through
gap 512 against a cold control at 0.475–0.54 (`beacon_swap.json`).

The 40-hour numbers (ratio 1.0091, 4.7430 vs 4.7782 from 8.6588, 25.17% gradient
tokens, twin excess 0.0029, injection −0.087 vs −0.279) are **unchanged** and were
re-verified against `results/pos_summary.json` / `pos_metrics.jsonl` conventions
as reported in v0.1.

## 4. The kept negative that entered the ledger section

`results/rank_sweep_final.json` (P34, relational rank) is now stated in the
falsifications subsection with its numbers: attention control solves every load
(test recall 0.9888 → 0.9997 mean over 4 seeds; min single-seed 0.9819),
both state arms at chance from K=4 (chance 0.015625), cliff ratio **1.0** against
a registered ≥2, one eligible cell at K=2 with ratio **0.371** — the opposite
sign of the prediction — and the (a′) check reported as **"insufficient data"**
rather than as a pass or a fail (no load clears 3× chance for phase with the
required ≥2 ignited seeds). The dominating phenomenon — phase ignition collapsing
with load, 3/4 ignited seeds at K=2 to 0/4 by K=16 — is reported as the finding.

> Note: `rank_sweep_final.json` is the complete grid (K ∈ {2,4,8,16,32}) and is the
> ledger's own cited anchor. `results/rank_sweep.json` holds the same run without the
> K=32 cell; v0.2 cites the final file.

## 5. References

`o1state_refs.bib` unchanged (13 entries). Two entries present but uncited in
v0.1 are now cited in the collective-memory section: `vandeven2019replay`
(replay as a continual-learning mechanism) and `kirkpatrick2017ewc` (the
weight-space penalty this design does *not* use). The related-work note lost its
"to be expanded in v0.2" marker.

## 6. Files changed

- `paper/o1state.tex` — rewritten to v0.2
- `paper/CHANGES_v02.md` — this file (new)
- `src/plot_v02_figures.py` — new figure script
- `paper/figures/fig_gate_width_curve.pdf`, `fig_f2_sweep.pdf`,
  `fig_length_reanchor.pdf` — new
- `paper/o1state.pdf` + build artifacts — regenerated

Compile: `latexmk -pdf o1state` — **warning-free**, 14 pages.
