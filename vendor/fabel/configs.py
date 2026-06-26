"""
configs.py — the use-case registry for the fabel substrate.

ONE place that holds every operating mode: which graph/corpus, which extraction and
amplification knobs, which entry point. Every script, the eval harness, and the
dashboard read profiles from here, so a use-case is configured once and reused
everywhere. Add a new use-case = add one PROFILES entry.

A profile is a plain dict (no YAML/parse dependency, importable directly):
  kind         : what the profile does (hypotheses|controversy|memory|skill|answer|extract)
  description  : one line, shown in the dashboard and docs
  graph        : path to a .causal graph (for graph-consuming modes), or None
  corpus       : path to a paper directory (for extraction modes), or None
  concept_mode : True  -> concept head-NP entities (cross-paper bridging on)
  canonicalize : True  -> WordNet collapse so the graph connects (precondition for ranking)
  top          : how many results to surface
  params       : mode-specific knobs (tau floor, sinkhorn, rounds, ...)
"""
from __future__ import annotations

import os
from typing import Dict, Any

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPHS = os.path.join(HERE, "graphs")

# Default corpora live under /tmp from this session's builds; override via env or
# by editing the profile. They are documented as "build-it-first" in CONFIGS.md.
MED_CORPUS = os.environ.get("FABEL_MED_CORPUS", "/tmp/medcorpus_big")
MED_GRAPH = os.environ.get("FABEL_MED_GRAPH", "/tmp/med.causal")
SESSION_GRAPH = os.environ.get(
    "FABEL_SESSION_GRAPH",
    os.path.join(HERE, "..", "memory_graph", "all_sessions_v2.causal"))


PROFILES: Dict[str, Dict[str, Any]] = {

    "medical-hypotheses": {
        "kind": "hypotheses",
        "description": "Cross-paper causal leads from Cochrane reviews: A→B in "
                       "paper X + B→C in paper Y ⇒ A→C stated by neither.",
        "graph": None,
        "corpus": MED_CORPUS,
        "concept_mode": True,     # without this, no cross-paper bridge ever forms
        "canonicalize": True,     # connects the graph so Fiedler ranking is real
        "top": 15,
        "params": {"min_conf": 0.4, "rank": "fiedler"},
    },

    "controversy": {
        "kind": "controversy",
        "description": "Where the literature DISAGREES: opposing causal claims of "
                       "comparable strength from DIFFERENT papers (cross-paper gate).",
        "graph": None,
        "corpus": MED_CORPUS,
        "concept_mode": True,
        "canonicalize": True,
        "top": 10,
        "params": {"min_score": 0.05, "cross_paper": True},
    },

    "session-memory": {
        "kind": "memory",
        "description": "Deterministic recall over thousands of past Claude sessions "
                       "distilled to one graph (intents, entities, recency).",
        "graph": SESSION_GRAPH,
        "corpus": None,
        "concept_mode": True,
        "canonicalize": False,    # conversational substrate; DF-filter handles noise
        "top": 8,
        "params": {"df_ceiling": 0.15},
    },

    "skill-projection": {
        "kind": "skill",
        "description": "Project a .causal graph into a self-contained skill .md with "
                       "its inference closure pre-baked (build-once, read-as-skill).",
        "graph": MED_GRAPH,
        "corpus": None,
        "concept_mode": True,
        "canonicalize": False,
        "top": 40,
        "params": {"include_inferred": True},
    },

    "answer": {
        "kind": "answer",
        "description": "Conversational Q&A over a mounted causal graph with full "
                       "provenance and zero hallucination (fabel.py).",
        "graph": MED_GRAPH,
        "corpus": None,
        "concept_mode": True,
        "canonicalize": False,
        "top": 6,
        "params": {},
    },

    "extract-eval": {
        "kind": "extract",
        "description": "The deterministic extractor measured against the gold set "
                       "(held-out F1, zero false positives).",
        "graph": None,
        "corpus": None,           # uses eval/gold
        "concept_mode": False,    # gold path is the tuned, opt-in-off configuration
        "canonicalize": False,
        "top": 0,
        "params": {},
    },
}


def get(name: str) -> Dict[str, Any]:
    """Return a profile by name, or raise with the list of valid names."""
    if name not in PROFILES:
        raise KeyError(f"unknown profile '{name}'. "
                       f"Available: {', '.join(sorted(PROFILES))}")
    return PROFILES[name]


def names() -> list:
    return sorted(PROFILES)


def resolve_graph(profile: Dict[str, Any]) -> str:
    """Fall back to a shipped demo graph if the profile's graph is missing, so the
    dashboard/demo never hard-fails on a fresh checkout."""
    g = profile.get("graph")
    if g and os.path.exists(g):
        return g
    for fallback in ("smoking_persisted.causal", "base.causal"):
        p = os.path.join(GRAPHS, fallback)
        if os.path.exists(p):
            return p
    return g or ""


if __name__ == "__main__":
    import json
    for n in names():
        p = PROFILES[n]
        print(f"\n## {n}  [{p['kind']}]")
        print(f"   {p['description']}")
        src = p.get("graph") or p.get("corpus") or "eval/gold"
        print(f"   source: {src}")
        print(f"   concept_mode={p['concept_mode']} canonicalize={p['canonicalize']} "
              f"top={p['top']} params={json.dumps(p['params'])}")
