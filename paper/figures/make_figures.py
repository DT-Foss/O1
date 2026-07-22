#!/usr/bin/env python3
"""Regenerate all paper figures as vector PDFs from the result JSONs.

Every number plotted here is read from a result JSON in results/ or
paper/evidence_companion/ — nothing is hard-coded. One consistent style:
no in-figure titles (conclusions live in captions), restrained palette,
thin spines, vector output.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

ROOT = Path(__file__).resolve().parent.parent.parent   # repo root
RES = ROOT / "results"
EVC = ROOT / "paper" / "evidence_companion"
OUT = Path(__file__).resolve().parent
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- style
BLUE = "#0072B2"     # selective scan (the operator)
LBLUE = "#56B4E9"    # selective + PE / secondary selective
VERM = "#D55E00"     # pure / boundary attractor
GREEN = "#009E73"    # hero arms (NoPE, SSAS, full stack)
GRAY = "#5A5A5A"     # attention / transformer / neutral
LGRAY = "#B0B0B0"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
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
    plt.close(fig)
    print(f"  wrote {name}.pdf")


# =============================================================== fig 1
# Kernel unification: (a) family reduction errors, (b) match vs control
def fig_kernel():
    fam = jload(RES / "ssm_family_reduction_results.json")
    cg = jload(RES / "constant_gate_kernel_match_width_results.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.45),
                                   gridspec_kw={"wspace": 0.34})

    # --- (a) seq-vs-scan max|delta| per family member
    names, errs = [], []
    for c in fam["cases"]:
        names.append(c["name"].replace(" / ", "/"))
        errs.append(c["max_abs_err_seq_vs_parallel"])
    ypos = range(len(names))[::-1]
    colors = [BLUE if "GSSM" in n else GRAY for n in names]
    ax1.barh(list(ypos), errs, color=colors, height=0.62, zorder=3)
    ax1.set_xscale("log")
    ax1.set_xlim(1e-16 * 0.6, 1e-14)
    ax1.set_yticks(list(ypos))
    ax1.set_yticklabels(names)
    ax1.set_xlabel(r"max $|\Delta|$, sequential vs. parallel scan")
    eps64 = 2.220446049250313e-16
    ax1.axvline(eps64, color=GRAY, lw=0.7, ls=(0, (4, 3)), zorder=2)
    ax1.text(eps64 * 1.1, 3.42, r"float64 $\varepsilon$",
             fontsize=6.6, color=GRAY, ha="left", va="top")
    for y, e in zip(ypos, errs):
        ax1.text(e * 1.25, y, f"{e:.2e}", va="center", fontsize=6.8,
                 color="#333333")
    despine(ax1)
    ax1.grid(axis="x", linewidth=0.4, alpha=0.28, zorder=0)
    ax1.set_axisbelow(True)

    # --- (b) constant-gate kernel match vs selective control across widths
    widths = [128, 256, 512]
    match, ctrl = [], []
    for w in widths:
        pw = cg["per_width"][str(w)]
        match.append(max(l["max_abs_err"] for l in pw["kernel_match_constant_gate"]))
        ctrl.append(pw["selective_negative_control"]["max_abs_err"])
    ax2.plot(widths, ctrl, "s-", color=VERM, ms=4, lw=1.2, zorder=3,
             label="selective control (no single kernel)")
    ax2.plot(widths, match, "o-", color=BLUE, ms=4, lw=1.2, zorder=3,
             label=r"constant gate $=\gamma^{|s-t|}$ kernel")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-16, 1e0)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(widths)
    ax2.set_xticklabels([str(w) for w in widths])
    ax2.xaxis.set_minor_locator(FixedLocator([]))
    ax2.set_xlabel(r"width $d$")
    ax2.set_ylabel(r"max $|\Delta|$ vs. closed-form kernel")
    # bracket the separation at d=512
    ax2.annotate("", xy=(512, match[-1] * 3), xytext=(512, ctrl[-1] / 3),
                 arrowprops=dict(arrowstyle="<->", lw=0.7, color="#333333"))
    ax2.text(430, 3e-9, "13 orders\nof magnitude", fontsize=6.8,
             ha="right", va="center", color="#333333")
    ax2.legend(loc="center left", bbox_to_anchor=(0.02, 0.30), handlelength=1.6)
    despine(ax2)
    ygrid(ax2)

    for ax, lab in ((ax1, "A"), (ax2, "B")):
        ax.text(-0.16, 1.04, lab, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="top")
    save(fig, "fig_kernel_unification")


# =============================================================== fig 2
# Length extrapolation: PPL vs eval T, four arms
def fig_length():
    d = jload(RES / "length_extrap_v2.json")
    arms = [
        ("selective_nope", "Selective-NoPE", GREEN, "o", "-", 1.7),
        ("selective", "Selective+PE", BLUE, "o", "-", 1.2),
        ("transformer", "Transformer", GRAY, "^", "-", 1.2),
        ("pure", "Pure (no gate)", VERM, "s", "-", 1.2),
    ]
    fig, ax = plt.subplots(figsize=(5.3, 3.0))
    Ts = sorted(int(k) for k in d["selective"]["curve"] if k.isdigit())
    for key, label, color, mk, ls, lw in arms:
        ppl = [d[key]["curve"][str(t)]["ppl"] for t in Ts]
        drift = d[key]["curve"]["_delta_128_1024_pct"]
        ax.plot(Ts, ppl, marker=mk, ms=3.4, lw=lw, ls=ls, color=color,
                label=label, zorder=3)
        ax.annotate(f"{drift:+.1f}%", xy=(Ts[-1], ppl[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=6.8, color=color, va="center")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(Ts)
    ax.set_xticklabels([str(t) for t in Ts])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([200, 500, 1000, 2000, 5000])
    ax.set_yticklabels(["200", "500", "1000", "2000", "5000"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(Ts[0] * 0.92, Ts[-1] * 1.55)
    ax.axvline(32, color=GRAY, lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.text(30.6, 900, "train $T{=}32$", rotation=90, fontsize=6.8,
            color=GRAY, ha="center", va="center")
    ax.set_xlabel(r"evaluation length $T$ (tokens)")
    ax.set_ylabel("validation perplexity")
    ax.legend(loc="upper left", handlelength=1.8)
    despine(ax)
    ygrid(ax)
    save(fig, "fig_length_extrap_v2")


# =============================================================== fig 3
# Scale-up grid: best PPL across d x L
def fig_scaleup():
    d = jload(EVC / "scaleup.json")
    widths = [256, 512, 768, 1024]
    layers = [2, 4]
    fig, ax = plt.subplots(figsize=(3.25, 2.5))
    bw = 0.36
    best = min(v["best_ppl"] for v in d.values() if "best_ppl" in v)
    for j, L in enumerate(layers):
        xs, ys = [], []
        for i, w in enumerate(widths):
            key = f"selective_d{w}_L{L}_T128"
            entry = d[key]
            x = i + (j - 0.5) * bw
            if "best_ppl" in entry:
                xs.append(x)
                ys.append(entry["best_ppl"])
            else:  # skipped: RAM
                ax.bar(x, best, width=bw, facecolor="none",
                       edgecolor=LGRAY, lw=0.7, hatch="///", zorder=2)
                ax.text(x, best / 2, "skipped\n(RAM)", ha="center",
                        va="center", fontsize=5.8, color=GRAY, rotation=90)
        color = BLUE if L == 2 else GREEN
        ax.bar(xs, ys, width=bw, color=color, zorder=3, label=f"$L={L}$")
        for x, y in zip(xs, ys):
            ax.text(x, y + 1.2, f"{y:.0f}", ha="center", fontsize=6.2,
                    color="#333333")
    ax.axhline(best, color=GRAY, lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.text(-0.62, best - 4.5, f"best {best:.1f}", fontsize=6.5,
            color=GRAY, ha="left", va="top")
    ax.set_xticks(range(len(widths)))
    ax.set_xticklabels([f"$d={w}$" for w in widths])
    ax.set_ylabel("best validation perplexity")
    ax.set_ylim(0, 165)
    ax.legend(loc="lower right", ncols=2, handlelength=1.2,
              columnspacing=0.9, bbox_to_anchor=(1.0, 1.0))
    despine(ax)
    ygrid(ax)
    save(fig, "fig_scaleup")


# =============================================================== fig 4
# Saturation profile: interior vs boundary attractor (final epoch, d512)
def fig_saturation():
    # Phase-1 (unstabilized) runs at d128/d256 — the clean attractor read.
    # The d512 phase-1 arms carry the width-collapse pathology (see M1) and
    # are excluded here; the operator-level contrast lives at d128/d256.
    p = jload(EVC / "phase1_fulldata.json")
    pos_keys = ["first", "q1", "mid", "last"]
    pos_x = [0.0, 0.25, 0.5, 1.0]
    pos_lab = ["first", "25%", "mid", "last"]
    fig, ax = plt.subplots(figsize=(3.25, 2.5))
    series = [
        ("pure_d256_T32", "Pure $d{=}256$", VERM, "-", 1.5, 4.0),
        ("pure_d128_T32", "Pure $d{=}128$", VERM, (0, (4, 2)), 1.0, 2.8),
        ("selective_d256_T32", "Selective $d{=}256$", BLUE, "-", 1.5, 4.0),
        ("selective_d128_T32", "Selective $d{=}128$", BLUE, (0, (4, 2)),
         1.0, 2.8),
    ]
    for key, label, color, ls, lw, ms in series:
        prof = p[key]["gate_trace"][-1]["sat_profile"]
        ys = [prof[k] for k in pos_keys]
        ax.plot(pos_x, ys, marker="o", ls=ls, color=color, ms=ms, lw=lw,
                label=label, zorder=3)
    ax.annotate("boundary pull", xy=(0.99, 0.505), fontsize=6.6, color=VERM,
                ha="right", va="bottom")
    ax.annotate("interior: flat", xy=(0.99, 0.095), fontsize=6.6, color=BLUE,
                ha="right", va="bottom")
    ax.set_xticks(pos_x)
    ax.set_xticklabels(pos_lab)
    ax.set_xlabel("position in sequence")
    ax.set_ylabel(r"saturated fraction ($s_t > 0.95$)")
    ax.set_ylim(0, 0.62)
    ax.set_xlim(-0.05, 1.05)
    ax.legend(loc="upper left", handlelength=1.9, fontsize=6.6)
    despine(ax)
    ygrid(ax)
    save(fig, "fig_saturation")


# =============================================================== fig 5+6
# Double dissociation: recall bars and length curves
ARMS = [
    ("attn4", "AAAA", GRAY),
    ("hyb_mid", "SSAS", GREEN),
    ("hyb_top", "SSSA", BLUE),
    ("sel4", "SSSS", LBLUE),
    ("pure_proxy", "PPAP", VERM),
]


def fig_hybrid_recall():
    b = jload(EVC / "hybrid_B.json")
    fig, ax = plt.subplots(figsize=(3.25, 2.55))
    xs = range(len(ARMS))
    for i, (key, spec, color) in enumerate(ARMS):
        v = b[key]["mqar"]["test_len"]["overall"]
        ax.bar(i, v, width=0.62, color=color, zorder=3)
        ax.text(i, v + 0.022, f"{v:.3f}",
                ha="center", fontsize=6.6, color="#333333")
    chance = 1 / 64
    ax.axhline(chance, color=GRAY, lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.text(len(ARMS) - 0.5, chance + 0.015, "chance $1/64$", fontsize=6.3,
            color=GRAY, ha="right")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{k}\n({s})" for k, s, _ in ARMS], fontsize=6.6)
    ax.set_ylabel("MQAR recall (eval len 256)")
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    despine(ax)
    ygrid(ax)
    save(fig, "fig_hybrid_recall")


def fig_hybrid_length():
    a = jload(EVC / "hybrid_A.json")
    fig, ax = plt.subplots(figsize=(3.25, 2.55))
    Ts = sorted(int(k) for k in a["sel4"]["length_curve"] if k.isdigit())
    for key, spec, color in ARMS:
        ppl = [a[key]["length_curve"][str(t)]["ppl"] for t in Ts]
        rise = a[key]["rise_pct_lo_to_hi"]
        ls = (0, (4, 2)) if key == "attn4" else "-"
        ax.plot(Ts, ppl, "o", ls=ls, color=color, ms=3.0, lw=1.2,
                label=f"{spec}  ({rise:+.1f}%)", zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(Ts)
    ax.set_xticklabels([str(t) for t in Ts])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([200, 400, 800])
    ax.set_yticklabels(["200", "400", "800"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(Ts[0] * 0.93, Ts[-1] * 1.12)
    ax.axvline(32, color=GRAY, lw=0.6, ls=(0, (4, 3)), zorder=1)
    ax.set_xlabel(r"evaluation length $T$ (tokens)")
    ax.set_ylabel("validation perplexity")
    ax.legend(loc="upper left", handlelength=1.5, fontsize=6.6)
    despine(ax)
    ygrid(ax)
    save(fig, "fig_hybrid_length")


# =============================================================== fig 7
# Width fix: the d512 collapse and its resolution
WF_ARMS = [
    ("selective_baseline_d512_T32", "baseline", LGRAY),
    ("selective_clip_d512_T32", "clip", LGRAY),
    ("selective_warmup_d512_T32", "warmup", LBLUE),
    ("selective_preln_d512_T32", "preLN", LBLUE),
    ("selective_mup_d512_T32", r"$\mu$P", BLUE),
    ("selective_stack_d512_T32", "full stack", GREEN),
]


def fig_widthfix():
    w = jload(EVC / "width_fix.json")
    fig, ax = plt.subplots(figsize=(3.25, 2.5))
    for i, (key, label, color) in enumerate(WF_ARMS):
        v = w[key]["best_ppl"]
        ax.bar(i, v, width=0.62, color=color, zorder=3)
        ax.text(i, v + 2.5, f"{v:.0f}", ha="center", fontsize=6.6,
                color="#333333")
    ref = w["selective_stack_d128_d128_T32"]["best_ppl"]
    ax.axhline(ref, color=GRAY, lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.text(1.62, ref - 5, f"$d{{=}}128$ stack ({ref:.0f})",
            fontsize=6.3, color=GRAY, ha="center", va="top")
    ax.set_xticks(range(len(WF_ARMS)))
    ax.set_xticklabels([lab for _, lab, _ in WF_ARMS], fontsize=6.8,
                       rotation=20, ha="right")
    ax.set_ylabel("best validation perplexity")
    ax.set_ylim(0, 232)
    despine(ax)
    ygrid(ax)
    save(fig, "fig_widthfix")


# =============================================================== fig 8
# Attribution: velocity saturation and W_v preactivation per arm
def fig_attribution():
    w = jload(EVC / "width_fix.json")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.25, 2.7), sharex=True,
                                   gridspec_kw={"hspace": 0.18})
    xs = range(len(WF_ARMS))
    sat = [w[k]["velocity_metrics"]["velocity_sat_gt95"] for k, _, _ in WF_ARMS]
    pre = [w[k]["velocity_metrics"]["wv_preact_absmax"] for k, _, _ in WF_ARMS]
    colors = [c for _, _, c in WF_ARMS]
    ax1.bar(xs, sat, width=0.62, color=colors, zorder=3)
    for x, v in zip(xs, sat):
        ax1.text(x, v + 0.012, f"{v:.2f}", ha="center", fontsize=6.0,
                 color="#333333")
    ax1.set_ylabel("velocity sat.", fontsize=7.5)
    ax1.set_ylim(0, 0.55)
    ax1.set_yticks([0, 0.2, 0.4])
    despine(ax1, keep=("left",))
    ygrid(ax1)
    ax2.bar(xs, pre, width=0.62, color=colors, zorder=3)
    for x, v in zip(xs, pre):
        ax2.text(x, v * 1.25, f"{v:.0f}", ha="center", fontsize=6.0,
                 color="#333333")
    ax2.set_yscale("log")
    ax2.set_ylim(5, 900)
    ax2.set_yticks([10, 100])
    ax2.set_yticklabels(["10", "100"])
    ax2.yaxis.set_minor_formatter(NullFormatter())
    ax2.set_ylabel(r"$\|W_v\|$ preact.", fontsize=7.5)
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels([lab for _, lab, _ in WF_ARMS], fontsize=6.8,
                        rotation=20, ha="right")
    despine(ax2)
    ygrid(ax2)
    save(fig, "fig_attribution")


if __name__ == "__main__":
    print("regenerating figures ->", OUT)
    fig_kernel()
    fig_length()
    fig_scaleup()
    fig_saturation()
    fig_hybrid_recall()
    fig_hybrid_length()
    fig_widthfix()
    fig_attribution()
    print("done.")
