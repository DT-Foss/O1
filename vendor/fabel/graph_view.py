"""
graph_view.py — the graph as fabel (and the model) works with it.

A binary triplet store read one fact at a time is opaque. But once every edge is
typed and the inference is materialized, the WHOLE reasoning surface is laid out
at once: explicit edges, the chains they amplify into, the conditions/populations
that gate each chain, the contradictions, the temporal flags, the gaps. This
renders that surface as text — the view an agent reasons over, not a UI.

`inside_view(causal_path)` loads a graph and prints:
  - explicit edges, each with its typed fields
  - typed 2-hop chains (with accumulated conditions / intersected populations /
    multiplied effect sizes), and which chains were PRUNED by a typed rule
  - contradictions (and which are resolved by disjoint populations)
  - temporal data-quality flags
  - hyperedges and their firing conditions
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "dotcausal_package", "src"))

from dotcausal import CausalReader
import typed_inference as ti


def load_edges(causal_path: str):
    """Read explicit edges with their typed attrs."""
    trips = CausalReader(causal_path).get_all_triplets(include_inferred=False)
    edges = []
    for t in trips:
        edges.append({
            "trigger": t["trigger"], "mechanism": t["mechanism"],
            "outcome": t["outcome"], "confidence": t.get("confidence", 0.7),
            "attrs": t.get("attrs", {}) or {},
        })
    return edges


def _attr_str(a: dict) -> str:
    bits = []
    if a.get("condition"):
        bits.append(f"when:{a['condition']}")
    if a.get("population"):
        bits.append(f"pop:{a['population']}")
    if a.get("effect_size"):
        bits.append(f"effect:{a['effect_size']}")
    if a.get("co_causes"):
        bits.append(f"joint:{'+'.join(a['co_causes'])}")
    if a.get("temporal"):
        bits.append(f"time:{a['temporal']}")
    return "  {" + ", ".join(bits) + "}" if bits else ""


def inside_view(causal_path: str) -> str:
    edges = load_edges(causal_path)
    lines = []
    lines.append(f"GRAPH: {os.path.basename(causal_path)} — "
                 f"{len(edges)} explicit edges")
    lines.append("")

    lines.append("EXPLICIT EDGES (with typed fields)")
    for e in edges:
        lines.append(f"  {e['trigger']} --[{e['mechanism']}]--> "
                     f"{e['outcome']}{_attr_str(e['attrs'])}")
    lines.append("")

    # typed chains + pruning count
    chains = ti.derive(edges)
    # count how many 2-hop opportunities existed vs survived (pruned by types)
    by_trig = {}
    for e in edges:
        by_trig.setdefault(e["trigger"].lower(), []).append(e)
    opportunities = sum(
        1 for ab in edges for bc in by_trig.get(ab["outcome"].lower(), [])
        if bc["outcome"].lower() != ab["trigger"].lower())
    pruned = opportunities - len(chains)
    lines.append(f"TYPED CHAINS (2-hop) — {len(chains)} derived, "
                 f"{pruned} pruned by typed rules")
    for c in chains:
        lines.append("  " + ti.explain(c))
    lines.append("")

    # contradictions
    conflicts = ti.contradictions(edges)
    if conflicts:
        lines.append("CONTRADICTIONS")
        for c in conflicts:
            mark = "REAL" if c["real_contradiction"] else "resolved"
            lines.append(f"  [{mark}] {c['entity']} -> {c['outcome']}: "
                         f"+{c['positive']} / -{c['negative']} ({c['note']})")
        lines.append("")

    # temporal flags
    viol = ti.temporal_violations(edges)
    if viol:
        lines.append("TEMPORAL DATA-QUALITY FLAGS")
        for v in viol:
            lines.append(f"  {v['trigger']} {v['mechanism']} {v['outcome']}: "
                         f"{v['issue']}")
        lines.append("")

    # hyperedges
    hyper = [e for e in edges if (e["attrs"] or {}).get("co_causes")]
    if hyper:
        lines.append("HYPEREDGES (joint causation — need all members)")
        for e in hyper:
            co = e["attrs"]["co_causes"]
            lines.append(f"  {{{', '.join(co)}}} --> {e['outcome']} "
                         f"(fires only when all {len(co)} present)")
        lines.append("")

    return "\n".join(lines)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        HERE, "graphs", "smoking_demo.causal")
    print(inside_view(path))


if __name__ == "__main__":
    main()
