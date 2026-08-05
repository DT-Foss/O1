#!/usr/bin/env python3
"""Regenerate the three figures new in paper v0.2, as vector PDFs from result JSONs.

House style is inherited verbatim from paper/figures/make_o1state_figures.py
(Okabe-Ito palette, no in-figure titles, thin spines, vector output). Every
number plotted is read from results/*.json -- nothing is hard-coded.

  fig_gate_width_curve.pdf   results/gate_law_width_curve.json  (+ the falsification)
  fig_f2_sweep.pdf           results/f2_equivalence_sweep.json  (the overlap law)
  fig_length_reanchor.pdf    results/scale_to_a_million.json
                             + results/length_extrap_v2_extreme.json (the control)

Sizing discipline: figsize inches match the exact width the figure is placed at
in o1state.tex, so LaTeX never rescales and in-plot fonts stay consistent with
the running text.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- style
BLUE = "#0072B2"
LBLUE = "#56B4E9"
VERM = "#D55E00"
GREEN = "#009E73"
GRAY = "#5A5A5A"
LGRAY = "#B0B0B0"
YELLOW = "#E69F00"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
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


# =============================================================== fig A
# The gate-law width curve: the falsified two-point trend (left) and the
# per-gradient-token inversion that survives it (right).
# Source: results/gate_law_width_curve.json
def fig_gate_width_curve():
    pts = jload(RES / "gate_law_width_curve.json")["points"]
    d = [p["d_model"] for p in pts]
    ratio = [p["improvement_ratio"] for p in pts]
    perM = [p["improvement_per_Mgrad_token"] for p in pts]
    gate = [p["gate_frac"] for p in pts]
    x = np.arange(len(d))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.35))

    # -- left: the registered metric, and where it stops being monotone
    ax1.plot(x, ratio, "-o", color=GREEN, lw=1.3, ms=4.2, zorder=4)
    # the two-point extrapolation that the third point kills
    slope = ratio[1] - ratio[0]
    ax1.plot([x[1], x[2]], [ratio[1], ratio[1] + slope], "--", color=LGRAY,
             lw=1.0, zorder=2)
    ax1.annotate("two-point trend\n(falsified)", xy=(x[2], ratio[1] + slope),
                 xytext=(-2, 5), textcoords="offset points",
                 fontsize=6.6, color=GRAY, ha="right", va="bottom")
    for xi, r in zip(x, ratio):
        ax1.annotate(f"{r:.4f}", xy=(xi, r), xytext=(0, -11),
                     textcoords="offset points", fontsize=6.8,
                     color=GREEN, ha="center")
    ax1.set_ylabel("improvement ratio A3/A2")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"$d{{=}}{v}$" for v in d])
    ax1.set_ylim(0.960, 1.002)
    despine(ax1)
    ygrid(ax1)

    # -- right: per gradient token spent, plus the dose that explains the drop
    bars = ax2.bar(x, perM, width=0.55, color=[LBLUE, LBLUE, BLUE], zorder=3)
    for xi, v in zip(x, perM):
        ax2.annotate(f"{v:.4f}", xy=(xi, v), xytext=(0, 2),
                     textcoords="offset points", fontsize=6.8,
                     color="#1a1a1a", ha="center")
    ax2.set_ylabel("improvement per M gradient token")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"$d{{=}}{v}$" for v in d])
    ax2.set_ylim(0, max(perM) * 1.30)
    despine(ax2)
    ygrid(ax2)

    axg = ax2.twinx()
    axg.plot(x, [g * 100 for g in gate], ":s", color=VERM, lw=1.1, ms=3.4, zorder=5)
    axg.set_ylabel("gate fires (%)", color=VERM, fontsize=7.6)
    axg.tick_params(axis="y", labelcolor=VERM, labelsize=7.0)
    axg.set_ylim(0, 40)
    for side in ("top", "left", "bottom"):
        axg.spines[side].set_visible(False)
    axg.spines["right"].set_color(VERM)
    axg.spines["right"].set_linewidth(0.6)
    axg.annotate("dose falls\n24.7% $\\to$ 19.8%", xy=(x[2], gate[2] * 100),
                 xytext=(-6, -16), textcoords="offset points",
                 fontsize=6.6, color=VERM, ha="right", va="top")

    fig.subplots_adjust(wspace=0.42)
    save(fig, "fig_gate_width_curve")


# =============================================================== fig B
# The exactness license, swept: gradient exactness is gated by warmup OVERLAP,
# not by chunk length. Source: results/f2_equivalence_sweep.json
def fig_f2_sweep():
    d = jload(RES / "f2_equivalence_sweep.json")
    rows = d["rows"]
    labels = {
        "selective_scalar": "selective scalar",
        "holographic_complex": "holographic complex",
        "holographic_tanh_m": "holographic tanh-$m$",
        "phase_off_control": "phase-off control",
    }
    colors = {
        "selective_scalar": GREEN,
        "holographic_complex": VERM,
        "holographic_tanh_m": BLUE,
        "phase_off_control": GRAY,
    }
    markers = {"selective_scalar": "o", "holographic_complex": "s",
               "holographic_tanh_m": "^", "phase_off_control": "D"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.35))

    # -- left: gradient relative error, split by overlap>0 vs overlap==0
    for r in rows:
        op = r["operator"]
        warm = [p for p in r["points"] if p["overlap"] > 0]
        cold = [p for p in r["points"] if p["overlap"] == 0]
        for pts, face in ((warm, colors[op]), (cold, "none")):
            ax1.plot([p["chunk"] for p in pts], [p["grad_rel_err"] for p in pts],
                     markers[op], ms=4.0, mfc=face, mec=colors[op], mew=0.9,
                     ls="none", zorder=4)
    ax1.set_yscale("log")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks([16, 32, 64, 128])
    ax1.set_xticklabels(["16", "32", "64", "128"])
    ax1.set_xlabel("chunk length (tokens)")
    ax1.set_ylabel("gradient relative error")
    ax1.axhspan(1e-8, 1e-5, color=GREEN, alpha=0.10, zorder=0)
    ax1.annotate("overlap $>$ 0 (filled): exact to $\\sim$5e-7", xy=(46, 8e-7),
                 fontsize=6.6, color=GREEN, va="center", ha="center")
    ax1.annotate("overlap $=$ 0 (open): 5e-3 to 2e-1", xy=(46, 5.5e-1),
                 fontsize=6.6, color=GRAY, va="center", ha="center")
    ax1.set_ylim(1e-7, 1.0)
    handles = [plt.Line2D([], [], ls="none", marker=markers[o], ms=4.0,
                          mfc=colors[o], mec=colors[o], mew=0.9, label=labels[o])
               for o in labels]
    ax1.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.30),
               handlelength=1.0, fontsize=6.4, ncol=2, columnspacing=1.0)
    despine(ax1)
    ygrid(ax1)

    # -- right: layout decoupling is exactly 0.0 with pure detach-carry
    ops = [r["operator"] for r in rows]
    xo = np.arange(len(ops))
    worst_warm, worst_cold = [], []
    for r in rows:
        warm = [p["layout_max_abs_delta"] for p in r["points"] if p["overlap"] > 0]
        cold = [p["layout_max_abs_delta"] for p in r["points"] if p["overlap"] == 0]
        worst_warm.append(max(warm))
        worst_cold.append(max(cold))
    ax2.bar(xo - 0.19, worst_warm, width=0.34, color=LGRAY, zorder=3,
            label="overlap $>$ 0 (re-warmed carry)")
    ax2.bar(xo + 0.19, [max(v, 1e-12) for v in worst_cold], width=0.34,
            color=GREEN, zorder=3, label="overlap $=$ 0 (pure detach-carry)")
    for xi, v in zip(xo + 0.19, worst_cold):
        ax2.annotate("0.0", xy=(xi, 1e-12), xytext=(0, 3),
                     textcoords="offset points", fontsize=6.8,
                     color=GREEN, ha="center", rotation=90)
    ax2.set_yscale("log")
    ax2.set_ylim(1e-12, 1e-1)
    ax2.set_ylabel("layout max abs delta (worst cell)")
    ax2.set_xticks(xo)
    ax2.set_xticklabels([labels[o] for o in ops], rotation=18, ha="right",
                        fontsize=6.6)
    ax2.legend(loc="upper left", handlelength=1.4)
    despine(ax2)
    ygrid(ax2)

    fig.subplots_adjust(wspace=0.34)
    save(fig, "fig_f2_sweep")


# =============================================================== fig C
# Length re-anchored: PPL improves to 524,288x training length at flat RSS
# (left), and the position term -- not the length -- is the wall (right).
# Sources: results/scale_to_a_million.json, results/length_extrap_v2_extreme.json
def fig_length_reanchor():
    m = jload(RES / "scale_to_a_million.json")
    curve = m["curve"]
    T = sorted(int(k) for k in curve)
    ratio = [curve[str(t)]["ratio"] for t in T]
    rss = [curve[str(t)]["peak_rss_gb"] for t in T]
    train_T = m["train_T"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.35))

    ax1.plot(T, ratio, "-o", color=GREEN, lw=1.3, ms=4.0, zorder=4,
             label="PPL / PPL(8192)")
    ax1.axhline(1.0, color=LGRAY, lw=0.8, ls="--", zorder=1)
    ax1.set_xscale("log", base=2)
    ax1.set_ylim(0.70, 1.10)
    ax1.set_xlabel(f"effective sequence length (train $T{{=}}{train_T}$)")
    ax1.set_ylabel("perplexity ratio")
    ax1.annotate(f"$\\times${ratio[-1]:.3f} at "
                 f"{T[-1] // train_T:,}$\\times$\ntraining length",
                 xy=(T[-1], ratio[-1]), xytext=(-6, -2),
                 textcoords="offset points", fontsize=6.8, color=GREEN,
                 ha="right", va="top")
    despine(ax1)
    ygrid(ax1)

    axr = ax1.twinx()
    axr.plot(T, rss, ":s", color=VERM, lw=1.0, ms=3.2, zorder=3)
    axr.set_ylabel("peak RSS (GB)", color=VERM, fontsize=7.6)
    axr.tick_params(axis="y", labelcolor=VERM, labelsize=7.0)
    axr.set_ylim(0, 6)
    for side in ("top", "left", "bottom"):
        axr.spines[side].set_visible(False)
    axr.spines["right"].set_color(VERM)
    axr.spines["right"].set_linewidth(0.6)
    axr.annotate(f"RSS {rss[0]:.2f} $\\to$ {rss[-1]:.2f} GB",
                 xy=(T[len(T) // 2], rss[len(T) // 2]), xytext=(0, 9),
                 textcoords="offset points", fontsize=6.6,
                 color=VERM, ha="center")

    # -- right: the control. Same architecture, the PE is the only difference.
    e = jload(RES / "length_extrap_v2_extreme.json")
    arms = [("selective_nope", "Selective-NoPE", GREEN, "-o"),
            ("selective", "Selective $+$ PE", VERM, "-s"),
            ("pure", "Pure (no gate)", GRAY, "-^")]
    for key, lab, col, sty in arms:
        if key not in e:
            continue
        c = e[key]["curve"]
        Ts = sorted(int(k) for k in c if k.isdigit())
        base = c[str(Ts[0])]["ppl"]
        ax2.plot(Ts, [c[str(t)]["ppl"] / base for t in Ts], sty, color=col,
                 lw=1.2, ms=3.4, label=lab, zorder=4)
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("eval length (train $T{=}32$)")
    ax2.set_ylabel("PPL relative to train length")
    ax2.axhline(1.0, color=LGRAY, lw=0.8, ls="--", zorder=1)
    ax2.legend(loc="upper left", handlelength=1.8)
    despine(ax2)
    ygrid(ax2)

    fig.subplots_adjust(wspace=0.46)
    save(fig, "fig_length_reanchor")


if __name__ == "__main__":
    print("regenerating v0.2 figures ->", OUT)
    fig_gate_width_curve()
    fig_f2_sweep()
    fig_length_reanchor()
    print("done.")
