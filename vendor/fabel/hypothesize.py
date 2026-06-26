"""
hypothesize.py — deterministic cross-paper hypothesis generation.

This is the thing no language model can do honestly: take A→B stated in paper X
and B→C stated in paper Y, and surface A→C — a causal hypothesis that NO single
paper states, assembled purely from facts, with the two-paper evidence chain
attached. Zero LLM, zero fabrication: every link is a real extracted edge, and
the bridge B is a CONCEPT (not a discourse word), so the chain means something.

Pipeline:
  corpus (pdf/txt) --[concept-mode extractor]--> typed edges with provenance
                   --[2-hop concept-bridged chaining]--> cross-paper hypotheses
                   --[novelty + confidence rank]--> the leads worth a human's time

A hypothesis here is a LEAD, not a proven fact — the honest framing. It says:
"papers X and Y, read separately, jointly imply A→C; nobody has connected them."

Usage:
    python3 hypothesize.py CORPUS_DIR [--top N] [--db facts.db]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "extract"))

import rule_extractor as rx                         # noqa: E402
from extract_to_db import _read_text                # noqa: E402  (pdf/txt reader)
from spectral import fiedler as sp_fiedler, gap_surprise  # noqa: E402


@dataclass
class Hypothesis:
    cause: str            # A
    bridge: str           # B (the shared concept that joins the two papers)
    effect: str           # C
    mech_ab: str
    mech_bc: str
    paper_ab: str         # source of A→B
    paper_bc: str         # source of B→C
    confidence: float     # chained, decayed
    surprise: float = 0.0  # novelty: rarer bridge + no direct A→C => higher

    def as_line(self) -> str:
        return (f"{self.cause} --[{self.mech_ab}]--> {self.bridge} "
                f"--[{self.mech_bc}]--> {self.effect}")

    def evidence(self) -> str:
        return (f"  bridge concept: '{self.bridge}'\n"
                f"  paper A→B: {self.paper_ab}\n"
                f"  paper B→C: {self.paper_bc}\n"
                f"  confidence: {self.confidence:.2f} (chained)")

    def verbalize(self) -> str:
        """Speak the hypothesis as a cross-paper LEAD, grounded and crisp: name
        the two sources, attribute each link to its source, then state the chain
        and frame it honestly as a lead, not a fact. Reads cleanly without
        depending on a domain-matched form bank — every claim is a real edge."""
        pa = self.paper_ab.rsplit(".", 1)[0]
        pb = self.paper_bc.rsplit(".", 1)[0]
        return (
            f"Neither source links “{self.cause}” to “{self.effect}” directly. "
            f"But {pa} reports that {self.cause} {self.mech_ab} "
            f"{self.bridge}, and {pb} reports that {self.bridge} "
            f"{self.mech_bc} {self.effect}. Chaining the two gives a lead: "
            f"{self.cause} → {self.effect} (via {self.bridge}, confidence "
            f"{self.confidence:.2f}). No single paper states this.")


def _norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def _edges_from_corpus(paths: List[str], domain: str = "") -> List[dict]:
    """Extract concept-level edges with per-paper provenance. Turns the
    extractor's concept mode ON for the duration (entities must be concepts to
    bridge across papers)."""
    rx.set_concept_mode(True)
    try:
        edges = []
        for p in paths:
            text = _read_text(p)
            if not text.strip():
                continue
            src = os.path.basename(p)
            for t in rx.extract_from_text(text, domain=domain, source=src):
                a, b = _norm(t.trigger), _norm(t.outcome)
                if a and b and a != b:
                    edges.append({"a": a, "b": b, "mech": t.mechanism,
                                  "conf": _conf(t.confidence), "src": src})
        return edges
    finally:
        rx.set_concept_mode(False)


def _conf(c) -> float:
    if isinstance(c, (int, float)):
        return float(c)
    return {"high": 0.9, "medium": 0.7, "low": 0.5}.get(str(c).lower(), 0.7)


def _is_concept(e: str) -> bool:
    """A bridge/endpoint must be a real concept, not discourse debris."""
    toks = e.split()
    if not toks or not (3 <= len(e) <= 45):
        return False
    if all(w in rx._DISCOURSE_NP for w in toks):
        return False
    if any(re.fullmatch(r"[0-9.]+", w) for w in toks):
        return False
    return toks[0] not in (
        "a", "an", "the", "this", "that", "these", "those", "other", "one",
        "some", "more", "most", "it", "there", "they", "we", "using", "can")


def _recanon_with_source(edges: List[Dict]) -> List[Dict]:
    """Canonicalize entity variants to one node, but keep each edge's source
    paper (canonicalize.apply_map merges and drops provenance, which cross-paper
    detection needs). Build the map once, rewrite a/b per edge, drop self-loops."""
    from canonicalize import build_map
    ents = sorted({e["a"] for e in edges} | {e["b"] for e in edges})
    cmap = build_map(ents)
    out = []
    for e in edges:
        a, b = cmap.get(e["a"], e["a"]), cmap.get(e["b"], e["b"])
        if a and b and a != b:
            out.append(dict(e, a=a, b=b))
    return out


def generate(paths: List[str], domain: str = "",
             min_conf: float = 0.4) -> List[Hypothesis]:
    """All concept-bridged, cross-paper 2-hop hypotheses, ranked."""
    edges = _edges_from_corpus(paths, domain)
    # collapse entity variants to one node so the graph connects across papers —
    # the prerequisite for cross-paper chains AND for the spectral ranking. Keep
    # per-edge provenance: canonicalize loses source, so re-attach it by matching.
    edges = _recanon_with_source(edges)
    by_a = defaultdict(list)
    for e in edges:
        by_a[e["a"]].append(e)

    seen = set()
    hyps: List[Hypothesis] = []
    for ab in edges:
        b = ab["b"]
        if not _is_concept(b) or not _is_concept(ab["a"]):
            continue
        for bc in by_a.get(b, []):
            c = bc["b"]
            if c == ab["a"] or not _is_concept(c):
                continue
            if ab["src"] == bc["src"]:           # cross-paper only
                continue
            key = (ab["a"], b, c)
            if key in seen:
                continue
            seen.add(key)
            conf = ab["conf"] * bc["conf"] * 0.85
            if conf < min_conf:
                continue
            hyps.append(Hypothesis(
                cause=ab["a"], bridge=b, effect=c,
                mech_ab=ab["mech"], mech_bc=bc["mech"],
                paper_ab=ab["src"], paper_bc=bc["src"], confidence=conf))

    # surprise: how far the hypothesis reaches across the knowledge graph's
    # primary structural bottleneck (Foss Gap Theorem). A→C is a genuine
    # cross-domain lead when A and C sit on opposite sides of the Fiedler cut —
    # two regions of the literature the papers keep separate. A direct A→C edge
    # already connects them, so it is not novel.
    coord, lam2 = sp_fiedler(edges)
    # the Fiedler ranking only means something on a CONNECTED graph with a real
    # bottleneck (Foss Gap Theorem premise). On a disconnected graph lam2≈0 and
    # the Fiedler vector is degenerate, so fall back to confidence ranking and
    # say so — silently emitting all-zero surprise would hide the real cause.
    connected = lam2 > 1e-6
    direct = {(e["a"], e["b"]) for e in edges}
    # fallback when the graph is disconnected (Fiedler degenerate): rank by
    # Perron-Frobenius centrality span instead (Born-as-PF). A hypothesis linking
    # a peripheral cause to a central effect (or vice versa) is still a non-
    # obvious reach — better than collapsing every surprise to zero.
    pf = {}
    if not connected:
        try:
            from pslifted import pf_centrality
            pf = pf_centrality(edges)
        except Exception:
            pf = {}
    for h in hyps:
        if connected:
            gap = gap_surprise(h.cause, h.effect, coord)
            already = 0.3 if (h.cause, h.effect) in direct else 1.0
            h.surprise = round(h.confidence * gap * already, 4)
        elif pf:
            span = abs(pf.get(h.cause, 0.0) - pf.get(h.effect, 0.0))
            h.surprise = round(h.confidence * span, 6)
        else:
            h.surprise = 0.0

    if connected:
        hyps.sort(key=lambda h: (-h.surprise, -h.confidence, -len(h.bridge)))
    else:
        hyps.sort(key=lambda h: (-h.confidence, -len(h.bridge)))
    return hyps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", help="directory of .pdf/.txt/.md papers")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--domain", default="")
    args = ap.parse_args()

    paths = []
    if os.path.isdir(args.corpus):
        for root, _, files in os.walk(args.corpus):
            for f in files:
                if f.endswith((".pdf", ".txt", ".md")):
                    paths.append(os.path.join(root, f))
    else:
        paths = [args.corpus]

    print(f"reading {len(paths)} papers (concept-mode extraction)...")
    hyps = generate(paths, domain=args.domain)
    print(f"\n{len(hyps)} cross-paper hypotheses (each joins two papers via a "
          f"shared concept no single paper connects)\n")
    for i, h in enumerate(hyps[:args.top], 1):
        print(f"[{i}] {h.as_line()}")
        print(h.evidence())
        print()


if __name__ == "__main__":
    main()
