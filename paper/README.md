# GSSM Unification Paper — build instructions

Source for the preprint *"A Kernel Unification of the Linear State-Space Family"*
(David Tom Foss, 2026).

## Files

- `gssm.tex` — the paper (neutral `article`-class preprint, no venue branding).
- `refs.bib` — bibliography; every entry checked against arXiv / proceedings metadata.
- Figures are pulled from `../plots/` via `\graphicspath` (no copies kept here).

## Build

Standard LaTeX toolchain (tested with TeX Live 2025 / TinyTeX, `pdflatex`):

```bash
cd paper
pdflatex gssm.tex
bibtex   gssm
pdflatex gssm.tex
pdflatex gssm.tex
```

or, equivalently, `latexmk -pdf gssm.tex`.

Produces `gssm.pdf` — **12 pages**, compiles clean: 0 overfull boxes, 0 undefined
references, 0 undefined citations. Only standard packages are used
(`amsmath`, `graphicx`, `booktabs`, `siunitx`, `subcaption`, `xcolor`, `hyperref`),
all present in a default TeX Live / TinyTeX install.

Bibliography style is `plain` (no `natbib` dependency).

## Figures used (from `../plots/`)

| file | claim |
|---|---|
| `fig_kernel_unification.png` | Kernel unification: family → one operator, Toeplitz LTI limit |
| `fig_length_extrap_v2.png`   | Length invariance: NoPE collapses drift +243% → +2.6% |
| `fig_scaleup.png`            | Stable scaling d256–d1024 × L2–L4, no collapse |
| `fig_saturation.png`         | Interior vs. boundary attractor |
| `fig_widthfix.png`           | μP recovers the d512 width collapse (202→153 PPL) |
| `fig_attribution.png`        | Root-cause: value projection hot at width, gates healthy |
| `fig_hybrid_recall.png`      | Double dissociation, Task B: SSAS 100% vs PPAP 16% recall |
| `fig_hybrid_length.png`      | Double dissociation, Task A: SSAS length-robust vs PPAP +313% |

## Number provenance

Every quantitative claim cites its result JSON. Sources fall in two groups.

**Measured in the O1 repository (`../results/`):**

- Kernel unification: `ssm_family_reduction_results.json`,
  `constant_gate_kernel_match_width_results.json`, `parallel_scan_integration_results.json`
- Length invariance: `length_extrap_v2.json`, `length_seed_robustness_d128.json`,
  `scale_to_the_wall.json`, `scale_wt103.json`, `scale_to_a_million.json`, `scale_to_a_billion.json`
- Holographic recall lift: `holographic_mqar.json`; rank theorem: `RANK1_CAPACITY_THEOREM.md`

**Measured in the companion GSSM architecture study, archived here in `evidence_companion/`:**

- Double dissociation (Task A + Task B): `hybrid_A.json`, `hybrid_B.json`
- Width-fix (M1): `width_fix.json`
- Scale-up (M3): `scaleup.json`, `scaleup_smoke.json`; phase-1 base scaling: `phase1_fulldata.json`

The six `evidence_companion/` JSONs are byte-identical across all four local GSSM-repo variants
(SHA-256 verified) and are copied here unmodified so the paper is self-contained. The double
dissociation is a **fully-evidenced claim**: SSAS 100% vs PPAP 16% MQAR recall (Task B), and
SSAS/sel4 length-robust (+35%/+39%) vs PPAP +313% (Task A), from `hybrid_A/B.json`.
