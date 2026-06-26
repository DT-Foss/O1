"""
controversy.py — find where the literature DISAGREES, deterministically.

Grounded in "Collapse Is Contraction" (Foss 2026): consensus dynamics CONTRACT
(τ<1) toward a single answer when the evidence agrees, but stay split (the τ→1,
non-contracting regime) when it doesn't. A controversy is exactly a place where
the local consensus does NOT contract — opposing causal claims of comparable
strength that no amount of averaging reconciles.

Made measurable without hand-waving:
  - Each causal edge has a POLARITY: positive (causes/increases) or negative
    (reduces/prevents/inhibits) — reusing typed_inference._neg.
  - For a target relation (cause→effect, or all causes of an effect), fuse the
    positive evidence and the negative evidence SEPARATELY via rapidity fusion
    (c = tanh(Σ arctanh cᵢ)) — the principled "N papers agreeing → →1" operator.
  - controversy score = 2·min(pos, neg)/(pos+neg) · min(pos,neg)  — high only when
    BOTH sides are independently strong (balanced AND confident). One-sided
    evidence (consensus) scores ~0.
  - CROSS-PAPER GATE: a controversy requires the two sides to come from DIFFERENT
    papers. One paper hedging itself is not a field-level disagreement.

Domain-agnostic: operates only on (trigger, mechanism, outcome, source). On a
drug-target or protein-interaction corpus, "activates vs. inhibits" / "binds vs.
does-not-bind" is the same polarity structure — the detector transfers unchanged.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from typed_inference import _neg               # polarity of a mechanism
from contraction import rapidity_fuse


@dataclass
class Controversy:
    cause: str
    effect: str
    pos_strength: float          # fused confidence of "increases/causes"
    neg_strength: float          # fused confidence of "reduces/prevents"
    score: float                 # balanced disagreement, high = real controversy
    pos_papers: List[str] = field(default_factory=list)
    neg_papers: List[str] = field(default_factory=list)
    pos_mechs: List[str] = field(default_factory=list)
    neg_mechs: List[str] = field(default_factory=list)

    def verbalize(self) -> str:
        pp = ", ".join(sorted(set(p.rsplit(".", 1)[0] for p in self.pos_papers))[:3])
        npp = ", ".join(sorted(set(p.rsplit(".", 1)[0] for p in self.neg_papers))[:3])
        return (
            f"The literature DISAGREES on whether “{self.cause}” affects "
            f"“{self.effect}”. Promoting evidence (strength {self.pos_strength:.2f}, "
            f"from {pp}): {', '.join(sorted(set(self.pos_mechs))[:2])}. "
            f"Opposing evidence (strength {self.neg_strength:.2f}, from {npp}): "
            f"{', '.join(sorted(set(self.neg_mechs))[:2])}. "
            f"No single direction is established — controversy score "
            f"{self.score:.2f}.")


def _conf(c) -> float:
    if isinstance(c, (int, float)):
        return float(c)
    return {"high": 0.9, "medium": 0.7, "low": 0.5}.get(str(c).lower(), 0.7)


def detect(edges: List[Dict], min_score: float = 0.05,
           cross_paper: bool = True) -> List[Controversy]:
    """Find (cause→effect) relations the corpus disagrees on. Each edge dict:
    {a, b, mech, conf, src}. Returns controversies ranked by score."""
    # group every claim by the (cause, effect) pair, split by polarity
    pairs: Dict[tuple, Dict[str, list]] = {}
    for e in edges:
        key = (e["a"], e["b"])
        slot = pairs.setdefault(key, {"pos": [], "neg": []})
        side = "neg" if _neg(e.get("mech", "")) else "pos"
        slot[side].append(e)

    out: List[Controversy] = []
    for (a, b), sides in pairs.items():
        pos, neg = sides["pos"], sides["neg"]
        if not pos or not neg:
            continue                       # one-sided → consensus, not controversy
        pos_papers = [e.get("src", "") for e in pos]
        neg_papers = [e.get("src", "") for e in neg]
        if cross_paper and not (set(pos_papers) - set(neg_papers) and
                                set(neg_papers) - set(pos_papers)):
            continue                       # same paper(s) both sides → self-hedge
        ps = rapidity_fuse([_conf(e.get("conf")) for e in pos])
        ns = rapidity_fuse([_conf(e.get("conf")) for e in neg])
        # balanced disagreement: high only when BOTH sides strong AND comparable
        balance = 2 * min(ps, ns) / (ps + ns) if (ps + ns) else 0.0
        score = balance * min(ps, ns)
        if score < min_score:
            continue
        out.append(Controversy(
            cause=a, effect=b, pos_strength=round(ps, 3),
            neg_strength=round(ns, 3), score=round(score, 3),
            pos_papers=pos_papers, neg_papers=neg_papers,
            pos_mechs=[e.get("mech", "") for e in pos],
            neg_mechs=[e.get("mech", "") for e in neg]))
    out.sort(key=lambda c: -c.score)
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    sys.path.insert(0, HERE)
    import hypothesize as H
    paths = []
    if os.path.isdir(args.corpus):
        for root, _, files in os.walk(args.corpus):
            for f in files:
                if f.endswith((".pdf", ".txt", ".md")):
                    paths.append(os.path.join(root, f))
    else:
        paths = [args.corpus]
    edges = H._recanon_with_source(H._edges_from_corpus(paths))
    cons = detect(edges)
    print(f"{len(cons)} controversies (cross-paper disagreements)\n")
    for i, c in enumerate(cons[:args.top], 1):
        print(f"[{i}] {c.verbalize()}\n")


if __name__ == "__main__":
    main()
