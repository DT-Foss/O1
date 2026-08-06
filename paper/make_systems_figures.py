#!/usr/bin/env python3
"""Figures unique to the systems paper (systems_v01.tex).

All data read from committed results/ artifacts. Two figures:
  fig_determinism.pdf  -- the 97,657-chunk determinism strip (det fields Δ=0,
                          wall-time is the only field that moves)
  fig_replication.pdf  -- the two-regime replication picture (chase debt
                          diverges, snapshot-sync erases it)

Run from paper/ :  python3 make_systems_figures.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def load_chunks(path, fields):
    """Stream a *_chunks.jsonl into {field: [values]}."""
    cols = {f: [] for f in fields}
    with open(path) as fh:
        for line in fh:
            j = json.loads(line)
            for f in fields:
                cols[f].append(j.get(f))
    return cols


def fig_determinism():
    f1 = os.path.join(RES, "pos_d512_chunks.jsonl")
    f2 = os.path.join(RES, "pos_rep50_d512_chunks.jsonl")
    fields = ["n", "g", "s1", "s2", "s3", "th", "w"]
    a = load_chunks(f1, fields)
    b = load_chunks(f2, fields)
    n = len(a["n"])

    det = ["n", "g", "s1", "s2", "s3", "th"]
    det_absdiff = []  # max abs diff over deterministic float fields per chunk
    w_diff = []
    for i in range(n):
        m = 0.0
        for fld in ["s1", "s2", "s3", "th"]:
            va, vb = a[fld][i], b[fld][i]
            if va is not None and vb is not None:
                m = max(m, abs(va - vb))
        # integer/bool fields
        if a["n"][i] != b["n"][i] or a["g"][i] != b["g"][i]:
            m = max(m, 1.0)
        det_absdiff.append(m)
        wa, wb = a["w"][i], b["w"][i]
        w_diff.append(abs((wa or 0.0) - (wb or 0.0)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 3.4), sharex=True)
    x = a["n"]

    # top: the deterministic fields — the S3 surprise trace of both runs overplotted
    ax1.plot(x, a["s3"], lw=0.5, color="#1f4e79", label="run 1 (original)")
    ax1.plot(x, b["s3"], lw=0.5, color="#d98a00", ls=(0, (3, 3)),
             label="run 2 (repeat, +11 days)")
    ax1.set_ylabel("held-out surprise $s_3$")
    ax1.legend(loc="upper right", fontsize=7, frameon=False)
    ax1.set_title(
        "Two runs, one recipe, 11 days apart: the deterministic record is bit-identical",
        fontsize=8.5)

    # bottom: per-chunk max abs delta on deterministic fields (== 0) vs wall-time delta
    ax2.plot(x, w_diff, lw=0.5, color="#999999",
             label="wall-clock time $|\\Delta w|$ (s)")
    ax2.plot(x, det_absdiff, lw=0.8, color="#1f4e79",
             label="max $|\\Delta|$ over $\\{n,g,s_1,s_2,s_3,\\theta\\}$")
    ax2.set_ylabel("per-chunk $|\\Delta|$")
    ax2.set_xlabel("chunk index (of 97{,}657)")
    ax2.legend(loc="upper right", fontsize=7, frameon=False)
    ax2.set_ylim(bottom=-0.1)

    maxdet = max(det_absdiff)
    ax2.text(0.02, 0.86,
             f"deterministic-field max $|\\Delta|$ over all 97{{,}}657 chunks = {maxdet:.1f}",
             transform=ax2.transAxes, fontsize=7, color="#1f4e79")

    fig.tight_layout()
    p = os.path.join(OUT, "fig_determinism.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p, "| deterministic max|Δ| =", maxdet,
          "| chunks with w-diff =", sum(1 for d in w_diff if d > 0))


def fig_replication():
    j = json.load(open(os.path.join(RES, "moebius_rate_check4.json")))
    cycles = j["cycles"]
    idx = [c["cycle"] for c in cycles]
    debt = j["p39d_scoring"]["debts_by_cycle"]
    catchup_s = [c["catchup_s"] for c in cycles]
    catchup_chunks = [c["catchup_chunks"] for c in cycles]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(5.5, 2.6))

    # left: standing sync-debt per cycle (chase regime, unequal pairing)
    axL.bar(idx, debt, color="#b03a2e", width=0.6)
    axL.set_xlabel("delta-sync cycle")
    axL.set_ylabel("standing sync-debt (chunks)")
    axL.set_title("Chase on an unequal pairing:\ndebt accrues", fontsize=8.5)
    axL.set_xticks(idx)
    for i, d in zip(idx, debt):
        axL.text(i, d + 400, f"{d:,}", ha="center", fontsize=7)
    axL.set_ylim(0, max(debt) * 1.25 + 1000)

    # right: catch-up time vs chunks replayed; cycle 0 is the 45k snapshot-sync replay
    axR.scatter(catchup_chunks, catchup_s, color="#1f4e79", zorder=3)
    for i, (cx, cs) in enumerate(zip(catchup_chunks, catchup_s)):
        axR.annotate(f"c{i}", (cx, cs), textcoords="offset points",
                     xytext=(5, 4), fontsize=7)
    axR.set_xlabel("chunks replayed in catch-up")
    axR.set_ylabel("catch-up wall time (s)")
    axR.set_title("Snapshot-sync replay:\n45k-chunk life in 458 s", fontsize=8.5)
    axR.grid(True, alpha=0.25)

    fig.tight_layout()
    p = os.path.join(OUT, "fig_replication.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p, "| debts =", debt, "| catchup_s =", catchup_s)


if __name__ == "__main__":
    fig_determinism()
    fig_replication()
