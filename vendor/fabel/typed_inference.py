"""
typed_inference.py — type-aware chaining. The PCR metaphor, made literal.

The plain .causal inference amplifies the EXISTENCE of a connection: A→B→C
becomes an inferred A→C. The risk, like PCR contamination, is amplifying noise:
long chains accumulate context that no longer applies.

Typed relations are the PRIMERS. They select which amplification is real and
they give the amplified product a richer readout than "it exists":

  condition  — a chain A→B (if Z) → C carries Z FORWARD. The inferred A→C holds
               only WHEN Z. Conditions accumulate along the chain (AND).
  population — A→B in P1, B→C in P2: the inferred A→C holds only in P1 ∩ P2.
               Disjoint populations BREAK the chain — no false amplification.
  effect_size— magnitudes MULTIPLY along a chain (HR 2.0 then HR 1.5 → ~3.0):
               the amplified product carries a quantitative titer, not just
               "connected".
  temporal   — if the chain's time order is impossible (effect precedes cause),
               the amplification is INVALID — a deterministic data-quality flag.

This module takes an explicit edge list (trigger, outcome, attrs) and derives
2-hop typed chains, propagating the typed fields by the rules above. Pure,
deterministic, no model.
"""
from __future__ import annotations

from typing import Dict, List, Optional


def _conf_to_num(c) -> float:
    if isinstance(c, (int, float)):
        return float(c)
    return {"high": 0.9, "medium": 0.7, "low": 0.5}.get(str(c).lower(), 0.7)


# generic population head-nouns that are NOT specific enough to prove overlap:
# "adult patients" and "pediatric patients" share "patients" but are DISJOINT.
# Only a shared SPECIFIC term (the modifier) counts as real overlap.
_GENERIC_POP = frozenset(
    "patients subjects people persons individuals participants cases adults "
    "children humans population populations group groups cohort cohorts "
    "men women".split())


def _populations_intersect(p1: Optional[str], p2: Optional[str]) -> Optional[str]:
    """Two population scopes. If both present and clearly disjoint -> None
    (chain breaks). If one is absent it doesn't constrain. Overlap requires a
    shared SPECIFIC term — a shared generic head ('patients', 'adults') does not
    count, so 'adult patients' vs 'pediatric patients' is correctly disjoint."""
    if not p1 or not p2:
        return p1 or p2
    t1, t2 = set(p1.lower().split()), set(p2.lower().split())
    # subset = one scope refines the other ('adults' ⊃ 'adults over 40'): real
    # overlap, even when the shared term is generic. Pick the more specific one.
    if t1 <= t2 or t2 <= t1:
        return p1 if len(t1) >= len(t2) else p2
    # otherwise overlap needs a shared SPECIFIC term; a shared generic head
    # ('patients', 'adults') alone does not count -> disjoint, chain breaks.
    if (t1 & t2) - _GENERIC_POP:
        return p1 if len(t1) >= len(t2) else p2
    return None


def _effect_product(e1: Optional[str], e2: Optional[str]) -> Optional[str]:
    """Multiply two effect magnitudes if both are 'X-fold' / 'ratio N' style."""
    import re

    def mag(e):
        m = re.search(r"(\d+(?:\.\d+)?)", e or "")
        return float(m.group(1)) if m else None
    m1, m2 = mag(e1), mag(e2)
    if m1 and m2:
        return f"~{round(m1 * m2, 2)} (combined)"
    return e1 or e2


def chain_two(edge_ab: Dict, edge_bc: Dict,
              decay: float = 0.85) -> Optional[Dict]:
    """Derive A→C from A→B and B→C, propagating typed fields. Returns the
    inferred edge dict, or None if a typed rule BREAKS the chain (disjoint
    populations) — which is the point: types prune false amplification.

    `decay` is the per-hop confidence contraction. The historical default 0.85 is
    a guess; `derive()` can pass the graph's MEASURED Dobrushin contraction
    coefficient instead (contraction.dobrushin) — the decay then reflects how
    fast THIS graph actually forgets, not a constant."""
    a_ab = edge_ab.get("attrs", {}) or {}
    a_bc = edge_bc.get("attrs", {}) or {}

    # population: intersect; disjoint breaks the chain
    pop = _populations_intersect(a_ab.get("population"), a_bc.get("population"))
    if a_ab.get("population") and a_bc.get("population") and pop is None:
        return None    # PRUNED: the two facts hold in disjoint populations

    # condition: accumulate (AND)
    conds = [c for c in (a_ab.get("condition"), a_bc.get("condition")) if c]
    condition = " and ".join(conds) if conds else None

    # effect size: multiply
    effect = _effect_product(a_ab.get("effect_size"), a_bc.get("effect_size"))

    # confidence: chain decay (measured contraction, or the historical default)
    conf = _conf_to_num(edge_ab.get("confidence")) * \
        _conf_to_num(edge_bc.get("confidence")) * decay

    attrs = {}
    if condition:
        attrs["condition"] = condition
    if pop:
        attrs["population"] = pop
    if effect:
        attrs["effect_size"] = effect

    return {
        "trigger": edge_ab["trigger"],
        "outcome": edge_bc["outcome"],
        "via": edge_ab["outcome"],
        "confidence": round(conf, 3),
        "is_inferred": True,
        "attrs": attrs,
        "rule": "typed-2hop",
    }


def derive(edges: List[Dict], decay: Optional[float] = None) -> List[Dict]:
    """All typed 2-hop chains over an edge list. Each edge: dict with trigger,
    outcome, confidence, attrs. Returns inferred edges; chains broken by a typed
    rule are omitted (and counted by the caller as pruned).

    If `decay` is None, the per-hop confidence contraction is MEASURED from this
    graph (Dobrushin coefficient) rather than assumed — so the inference decays at
    the rate the data actually mixes at. Pass a float to override."""
    if decay is None:
        try:
            from contraction import dobrushin
            decay = dobrushin([{"a": e["trigger"], "b": e["outcome"],
                                "conf": _conf_to_num(e.get("confidence"))}
                               for e in edges])
        except Exception:
            decay = 0.85
    decay = float(decay) if decay is not None else 0.85
    # index edges BY THEIR TRIGGER, so for A→B we find the B→C edges
    by_trigger: Dict[str, List[Dict]] = {}
    for e in edges:
        by_trigger.setdefault(e["trigger"].lower(), []).append(e)
    out = []
    for ab in edges:
        # the bridge entity B is ab's outcome; find edges whose TRIGGER is B
        for bc in by_trigger.get(ab["outcome"].lower(), []):
            if bc["outcome"].lower() == ab["trigger"].lower():
                continue   # no trivial cycles A→B→A
            chained = chain_two(ab, bc, decay=decay)
            if chained:
                out.append(chained)
    return out


def explain(inferred: Dict) -> str:
    """One-line natural rendering of a typed inferred chain."""
    a = inferred.get("attrs", {})
    s = f"{inferred['trigger']} → (via {inferred['via']}) → {inferred['outcome']}"
    tags = []
    if a.get("condition"):
        tags.append(f"only when {a['condition']}")
    if a.get("population"):
        tags.append(f"in {a['population']}")
    if a.get("effect_size"):
        tags.append(f"effect {a['effect_size']}")
    if tags:
        s += "  [" + "; ".join(tags) + "]"
    return s


# ============================================================================
# Four more typed-graph analyses (the strongest from the design menu)
# ============================================================================

def _neg(mech: str) -> bool:
    """Does the mechanism express a negative/preventive polarity?"""
    return bool(set(str(mech).lower().split()) & {
        "prevents", "reduces", "blocks", "inhibits", "suppresses",
        "decreases", "lowers", "impairs", "stops", "diminishes"})


def contradictions(edges: List[Dict]) -> List[Dict]:
    """(1) Find A→Y vs A→¬Y. With populations, a pair is only a REAL
    contradiction if the populations overlap; disjoint populations are two
    scoped facts, not a conflict. Returns conflicts with a 'real' flag."""
    pairs: Dict[tuple, List[Dict]] = {}
    for e in edges:
        pairs.setdefault((e["trigger"].lower(), e["outcome"].lower()),
                         []).append(e)
    out = []
    for (a, y), group in pairs.items():
        pos = [e for e in group if not _neg(e["mechanism"])]
        neg = [e for e in group if _neg(e["mechanism"])]
        for p in pos:
            for n in neg:
                pp = (p.get("attrs") or {}).get("population")
                np_ = (n.get("attrs") or {}).get("population")
                overlap = _populations_intersect(pp, np_)
                real = not (pp and np_ and overlap is None)
                out.append({
                    "entity": a, "outcome": y,
                    "positive": p["mechanism"], "negative": n["mechanism"],
                    "pos_population": pp, "neg_population": np_,
                    "real_contradiction": real,
                    "note": ("genuine conflict" if real else
                             "resolved: disjoint populations"),
                })
    return out


def temporal_violations(edges: List[Dict]) -> List[Dict]:
    """(2) Data-quality PCR test: a causal edge A→B whose temporal field says B
    happens BEFORE A is impossible (effect precedes cause). Flag it."""
    out = []
    for e in edges:
        temp = (e.get("attrs") or {}).get("temporal")
        if not temp:
            continue
        # the temporal field on a causal edge that says the OUTCOME precedes the
        # trigger is a contradiction in the data itself
        if temp in ("after", "following", "subsequently") and not _neg(e["mechanism"]):
            # "A causes B, B after A" — consistent, skip
            continue
        if temp in ("before", "prior to", "preceding"):
            out.append({
                "trigger": e["trigger"], "outcome": e["outcome"],
                "mechanism": e["mechanism"], "temporal": temp,
                "issue": "effect appears to precede cause — check the data",
            })
    return out


def hyperedge_check(edges: List[Dict], known_present: set) -> List[Dict]:
    """(3) Necessity: for a hyperedge {A,B,C}→D, D may be inferred ONLY if ALL
    co-causes are present. Given the set of entities known present, return each
    hyperedge with whether it fires and which members are missing."""
    out = []
    for e in edges:
        co = (e.get("attrs") or {}).get("co_causes")
        if not co or len(co) < 2:
            continue
        present = {c.lower() for c in known_present}
        missing = [c for c in co if c.lower() not in present]
        out.append({
            "co_causes": co, "outcome": e["outcome"],
            "fires": len(missing) == 0,
            "missing": missing,
            "note": ("all causes present — fires" if not missing
                     else f"blocked: missing {', '.join(missing)}"),
        })
    return out


def rank_causes(edges: List[Dict], outcome: str) -> List[Dict]:
    """(4) Strongest causes of `outcome`, ranked by effect size (magnitude),
    falling back to confidence when no effect size is present."""
    import re
    target = outcome.lower()
    scored = []
    for e in edges:
        if e["outcome"].lower() != target or _neg(e["mechanism"]):
            continue
        es = (e.get("attrs") or {}).get("effect_size")
        mag = None
        if es:
            m = re.search(r"(\d+(?:\.\d+)?)", es)
            mag = float(m.group(1)) if m else None
        scored.append({
            "cause": e["trigger"], "mechanism": e["mechanism"],
            "effect_size": es,
            "strength": mag if mag is not None else _conf_to_num(e["confidence"]),
            "by": "effect_size" if mag is not None else "confidence",
        })
    return sorted(scored, key=lambda x: -x["strength"])
