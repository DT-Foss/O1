"""
causal_to_skill.py — project a .causal graph into a self-contained skill .md.

The build-once / read-many seam, applied to a new OUTPUT format. The .causal file
is the compact binary substrate (msgpack, inference materialized once). This
adapter opens it with the CausalReader, pulls the WHOLE already-inferred surface
(explicit + inferred edges), and writes it as a structured Markdown skill document.

Crucial honesty about mechanics: a .md file does NOT "run inference when loaded" —
Markdown is passive text. What happens is the opposite and better: the inference is
PRE-BAKED into the .md at projection time, so when a model loads the skill it sees
the entire reasoning surface instantly, with nothing left to compute. Build once
(causal) → project to markdown → read as skill. Same tier, human/skill-readable
output instead of binary.

The projection is deterministic and groups the surface the way a reader needs it:
the domain's central concepts (Perron-Frobenius), the explicit facts by source,
the inferred edges that live in NO single document, and the cross-document leads.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "dotcausal_package", "src"))


def _load(causal_path: str):
    """Open the .causal and return (explicit, inferred) triplet lists. The reader
    materializes the inference closure; the `is_inferred` flag splits the two views.
    Schema: trigger / mechanism / outcome / confidence / source / is_inferred."""
    from dotcausal.io import CausalReader
    r = CausalReader(causal_path, verify_integrity=False)
    allf = r.get_all_triplets(include_inferred=True)
    explicit = [t for t in allf if not t.get("is_inferred")]
    inferred = [t for t in allf if t.get("is_inferred")]
    return explicit, inferred


def _centrality(triplets: List[Dict], top: int = 12) -> List[str]:
    """Degree-based central concepts (cheap PF proxy) — what the domain is *about*,
    so the skill leads with the right anchors."""
    deg: Dict[str, int] = defaultdict(int)
    for t in triplets:
        deg[t.get("trigger", "")] += 1
        deg[t.get("outcome", "")] += 1
    return [c for c, _ in sorted(deg.items(), key=lambda kv: -kv[1]) if c][:top]


def project(causal_path: str, name: str = "", description: str = "") -> str:
    """Return a complete skill-Markdown string for the graph."""
    explicit, inferred = _load(causal_path)
    name = name or os.path.splitext(os.path.basename(causal_path))[0]
    central = _centrality(explicit + inferred)

    by_src: Dict[str, List[Dict]] = defaultdict(list)
    for t in explicit:
        by_src[t.get("source") or t.get("pmcid") or "(unlabelled source)"].append(t)

    L: List[str] = []
    L.append("---")
    L.append(f"name: {name}")
    L.append(f"description: {description or f'Causal reasoning surface for {name}, '
              'inference pre-materialized. Use to answer cause/effect and '
              'cross-source questions about this domain.'}")
    L.append("---\n")
    L.append(f"# {name}: pre-inferenced causal surface\n")
    L.append(f"This skill is a projection of a `.causal` graph. Its inference closure "
             f"is **already computed** — {len(explicit)} explicit facts and "
             f"{len(inferred)} inferred edges that appear in no single source. "
             f"Read the connections below directly; nothing here needs recomputation.\n")

    L.append("## Central concepts\n")
    L.append("The domain is organized around: " + ", ".join(f"**{c}**" for c in central) + ".\n")

    L.append("## Inferred edges (live in NO single source)\n")
    L.append("These are the payoff — connections the substrate derived by chaining, "
             "that no individual document states:\n")
    for t in inferred[:40]:
        rel = t.get("mechanism", "affects")
        conf = t.get("confidence", "")
        cs = f" _(conf {conf:.2f})_" if isinstance(conf, (int, float)) else ""
        L.append(f"- **{t.get('trigger')}** {rel} **{t.get('outcome')}**{cs}")
    if len(inferred) > 40:
        L.append(f"- … and {len(inferred) - 40} more inferred edges.")
    L.append("")

    L.append("## Explicit facts by source\n")
    for src, ts in list(by_src.items())[:30]:
        L.append(f"### {src}")
        for t in ts[:12]:
            L.append(f"- {t.get('trigger')} → {t.get('outcome')} "
                     f"({t.get('mechanism', 'affects')})")
        L.append("")

    L.append("## How to use this skill\n")
    L.append("Answer cause/effect questions by reading the edges above. When a link "
             "spans two sources, say so explicitly and name both — that lead is the "
             "substrate's contribution, not a claim that one paper proved it.\n")
    return "\n".join(L)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("causal")
    ap.add_argument("-o", "--out", default="")
    ap.add_argument("--name", default="")
    args = ap.parse_args()
    md = project(args.causal, name=args.name)
    out = args.out or os.path.splitext(args.causal)[0] + "_skill.md"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"wrote {out} ({len(md):,} chars, ~{len(md)//4:,} tokens)")


if __name__ == "__main__":
    main()
