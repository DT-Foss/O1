#!/usr/bin/env python3
"""Regenerate the O(1)-State paper figures as vector PDFs from the result JSONs.

Same house style as make_figures.py (the GSSM-unification paper figures):
no in-figure titles (conclusions live in captions), restrained palette,
thin spines, vector output. Every number plotted here is read from a
result JSON in results/ — nothing is hard-coded.

Sizing discipline (top-venue practice used only as an aesthetic reference —
no image, figure, or data from any other paper is embedded here; every
number below comes from our own results/*.json): figsize is chosen in
inches to match the exact width each figure is placed at in o1state.tex
(\\textwidth = 6.5in at margin=1in, letterpaper, 11pt). \\includegraphics
then uses that same absolute width in inches, so LaTeX never rescales the
figure and in-plot font sizes stay visually consistent with the document's
running text — no font blow-up or shrink from a fractional
width=0.7\\textwidth-style scale.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).resolve().parent.parent.parent   # repo root
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- style
# Okabe-Ito colorblind-safe palette throughout.
BLUE = "#0072B2"     # A2 full-gradient / primary series
LBLUE = "#56B4E9"    # secondary series
VERM = "#D55E00"     # twin / contrast arm
GREEN = "#009E73"    # hero arm (A3 surprise-gated / the interaction arm)
GRAY = "#5A5A5A"      # neutral / reference
LGRAY = "#B0B0B0"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",   # sans mathtext, matches sans body font
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.2,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def jload(p):
    with open(p) as f:
        return json.load(f)


def jlload(p):
    with open(p) as f:
        return [json.loads(line) for line in f]


def despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def ygrid(ax):
    ax.grid(axis="y", linewidth=0.4, alpha=0.28, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    w_in = fig.get_size_inches()[0]
    plt.close(fig)
    print(f"  wrote {name}.pdf  (figsize width {w_in:.2f}in -- place with "
          f"\\includegraphics[width={w_in:.2f}in] for a 1:1, unscaled fit)")


# =============================================================== fig 1
# The 40-hour experiment: held-out loss (A2 full-gradient vs A3 gated)
# and the gate fraction, vs streamed tokens. Source: results/pos_metrics.jsonl
def fig_40h_gate():
    evals = [e for e in jlload(RES / "pos_metrics.jsonl") if e.get("type") == "eval"]
    ratio = jload(RES / "pos_summary.json")["ratio_A3_vs_A2_improvement"]
    n = [e["n_streamed"] / 1e6 for e in evals]
    a2 = [e["arms"]["A2"]["heldout"] for e in evals]
    a3 = [e["arms"]["A3"]["heldout"] for e in evals]
    gate = [e["arms"]["A3"].get("gate_frac_cum") for e in evals]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 3.6), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.10})
    ax1.plot(n, a2, color=BLUE, lw=1.2, label="A2 full-gradient (100% grad tokens)", zorder=3)
    ax1.plot(n, a3, color=GREEN, lw=1.4, label="A3 surprise-gated (25.2% grad tokens)", zorder=4)
    ax1.set_ylabel("held-out NLL")
    ax1.set_yscale("log")
    ax1.set_yticks([5, 6, 7, 8])
    ax1.set_yticklabels(["5", "6", "7", "8"])
    ax1.yaxis.set_minor_formatter(NullFormatter())
    ax1.annotate(f"ratio {ratio:.4f}\n(A3 beats A2)",
                 xy=(n[-1], a3[-1]), xytext=(-6, 14), textcoords="offset points",
                 fontsize=6.8, color=GREEN, ha="right", va="bottom")
    ax1.legend(loc="upper right", handlelength=1.8)
    despine(ax1)
    ygrid(ax1)

    ax2.plot(n, gate, color=GRAY, lw=1.0, zorder=3)
    ax2.set_ylabel("gate frac.\n(cum.)", fontsize=7.5)
    ax2.set_ylim(0, 1.0)
    ax2.set_yticks([0, 0.5, 1.0])
    ax2.set_xlabel("streamed tokens (millions)")
    despine(ax2)
    ygrid(ax2)

    for ax, lab in ((ax1, "A"), (ax2, "B")):
        ax.text(-0.10, 1.05, lab, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="top")
    save(fig, "fig_40h_gate_beats_firehose")


# =============================================================== fig 2
# The twin: A3 (never restarted) vs A3R (forked at T+24h with zero carried
# state), held-out loss around the fork. Source: results/pos_metrics.jsonl
def fig_twin_recovery():
    evals = [e for e in jlload(RES / "pos_metrics.jsonl") if e.get("type") == "eval"]
    summ = jload(RES / "pos_summary.json")
    fork_n = summ["twin"]["forked_at_n"]

    n = [e["n_streamed"] / 1e6 for e in evals]
    a3 = [e["arms"]["A3"]["heldout"] for e in evals]
    n_r, a3r = [], []
    for e in evals:
        if "A3R" in e["arms"]:
            n_r.append(e["n_streamed"] / 1e6)
            a3r.append(e["arms"]["A3R"]["heldout"])

    # zoom the window to just around the fork, where the warmup tax (or its
    # absence) is actually visible -- the full 40h y-range swamps it flat.
    lo_n = fork_n / 1e6 - 15
    hi_n = fork_n / 1e6 + 60
    n_zoom = [x for x in n if lo_n <= x <= hi_n]
    a3_zoom = [y for x, y in zip(n, a3) if lo_n <= x <= hi_n]

    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ax.plot(n_zoom, a3_zoom, color=GREEN, lw=1.4, marker="o", ms=3.0,
            label="A3 (never restarted)", zorder=3)
    ax.plot(n_r, a3r, color=VERM, lw=1.4, ls=(0, (4, 2)), marker="o", ms=3.0,
            label="A3R (twin, forked T+24h, zero carried state)", zorder=4)
    ax.axvline(fork_n / 1e6, color=GRAY, lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.text(fork_n / 1e6 - 1, max(a3_zoom + a3r),
            "fork", rotation=90, fontsize=6.8, color=GRAY, ha="right", va="top")
    ax.set_xlim(lo_n, hi_n)
    ax.set_xlabel("streamed tokens (millions)")
    ax.set_ylabel("held-out NLL")
    excess = summ["twin"]["surprise_excess_first_2h"]
    ax.annotate(f"online surprise: excess {excess:.4f},\nconverged 1 chunk after fork",
                fontsize=6.8, color=VERM, xy=(0.98, 0.94), xycoords="axes fraction",
                ha="right", va="top")
    ax.legend(loc="lower right", handlelength=1.8, fontsize=6.8)
    despine(ax)
    ygrid(ax)
    save(fig, "fig_twin_recovery")


# =============================================================== fig 3
# The persistence knee: recall accuracy vs silence gap G, for the baseline
# and the two theory-led eval-time levers (clamp, magnitude refresh) and
# their interaction (both jointly) -- the intervention that moves the knee
# furthest (F3). Source: results/holo_clamp_refresh.json
def fig_knee():
    d = jload(RES / "holo_clamp_refresh.json")
    sweep = d["sweep"]
    chance = d.get("chance", 0.0625)

    variants = [
        ("unclamped_norefresh", "baseline (neither lever)", GRAY),
        ("clamp_norefresh", "+ eval-time clamp", LBLUE),
        ("unclamped_refresh", "+ magnitude refresh", VERM),
        ("clamp_refresh", "+ both (interaction)", GREEN),
    ]
    Gs = sorted({int(k.split("|")[2][1:]) for k in sweep.keys()})
    seeds = sorted({int(k.split("|")[1][4:]) for k in sweep.keys()})

    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for key, label, color in variants:
        ys = []
        for G in Gs:
            vals = [sweep[f"{key}|seed{s}|G{G}"]["accuracy"]
                    for s in seeds if f"{key}|seed{s}|G{G}" in sweep]
            ys.append(sum(vals) / len(vals) if vals else None)
        xs = [G for G, y in zip(Gs, ys) if y is not None]
        yy = [y for y in ys if y is not None]
        lw = 1.6 if key == "clamp_refresh" else 1.1
        ax.plot(xs, yy, marker="o", ms=3.6, lw=lw, color=color, label=label, zorder=3)

    ax.axhline(chance, color=GRAY, lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.text(Gs[-1], chance + 0.02, "chance", fontsize=6.6, color=GRAY, ha="right", va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Gs)
    ax.set_xticklabels([str(G) for G in Gs])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("silence gap $G$ (tokens)")
    ax.set_ylabel("recall accuracy (seed mean)")
    ax.set_ylim(0, 0.62)
    ax.legend(loc="upper right", handlelength=1.8, fontsize=6.6)
    despine(ax)
    ygrid(ax)
    save(fig, "fig_persistence_knee")


# =============================================================== fig 4
# The rent map: 16-cell grid of (P_max, d_model) -> phase-advantage rent
# (percentage points). Source: results/holo_rent_map.json
def fig_rent_map():
    d = jload(RES / "holo_rent_map.json")
    cells = d["analysis"]["per_cell"]
    pmax_vals = sorted({v["P_max"] for v in cells.values()})
    d_vals = sorted({v["d_model"] for v in cells.values()})

    grid = np.full((len(pmax_vals), len(d_vals)), np.nan)
    for v in cells.values():
        i = pmax_vals.index(v["P_max"])
        j = d_vals.index(v["d_model"])
        grid[i, j] = v["rent_pp"]

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    vmax = float(np.nanmax(np.abs(grid)))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=-vmax, vmax=vmax,
                   aspect="auto", origin="lower")
    for i in range(len(pmax_vals)):
        for j in range(len(d_vals)):
            val = grid[i, j]
            if not np.isnan(val):
                txt_color = "white" if abs(val) / vmax > 0.72 else "#1a1a1a"
                ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                        fontsize=7.0, color=txt_color)
    ax.set_xticks(range(len(d_vals)))
    ax.set_xticklabels([f"$d{{=}}{d}$" for d in d_vals])
    ax.set_yticks(range(len(pmax_vals)))
    ax.set_yticklabels([f"$P_{{max}}{{=}}{p}$" for p in pmax_vals])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cbar.set_label("phase-advantage rent (pp)", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.8)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    save(fig, "fig_rent_map")


if __name__ == "__main__":
    print("regenerating O1-state figures ->", OUT)
    fig_40h_gate()
    fig_twin_recovery()
    fig_knee()
    fig_rent_map()
    print("done.")
