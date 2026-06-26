"""
canonicalize.py — collapse entity variants to one node, so the graph CONNECTS.

The measured wall: fabel builds a dust of disconnected components because the
same concept appears under many surface forms — "chronic stress", "psychological
stress", "stress" become three nodes, so a chain in paper A never meets a chain
in paper B. Every spectral lever (Fiedler ranking, PS-Lifted propagation) needs a
CONNECTED graph; this is the prerequisite.

Deterministic canonicalization, no model:
  1. lemmatize each token (WordNet morphy) and lowercase → "stresses"→"stress"
  2. drop a leading qualifier adjective if a bare head-noun form is also attested
     ("chronic stress" → "stress" when "stress" is itself a node)
  3. WordNet synonym collapse: map a one-word concept to the lexicographically
     smallest lemma of its dominant noun synset, so attested synonyms merge
     ("neoplasm"→"tumor" if both appear). Conservative: only collapses when the
     synset is shared, never guesses.

The canonical map is built FROM the corpus (data-driven): a variant collapses
only toward a form that ALSO appears in the graph, so we never invent nodes.
"""
from __future__ import annotations

import re
from typing import Dict, List, Set

try:
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet as wn
    _LEM = WordNetLemmatizer()
    _HAS_WN = True
except Exception:
    wn = None
    _HAS_WN = False


def _lemma_tokens(s: str) -> List[str]:
    toks = re.findall(r"[a-z0-9-]+", s.lower())
    if _HAS_WN:
        return [_LEM.lemmatize(t, "n") for t in toks]
    return [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in toks]


def _syn_rep(word: str, attested: Set[str]) -> str:
    """If `word` shares its dominant noun synset with another attested one-word
    concept, map both to the lexicographically smallest shared lemma. Conservative
    — only merges concepts that ALREADY appear in the graph."""
    if not _HAS_WN:
        return word
    syns = wn.synsets(word, pos="n") if wn is not None else []
    if not syns:
        return word
    lemmas = {l.name().replace("_", " ").lower() for l in syns[0].lemmas()}
    cohort = (lemmas & attested) | {word}
    return min(cohort)


def build_map(entities: List[str]) -> Dict[str, str]:
    """Build a canonical map over the corpus's actual entity set. Each entity →
    its canonical form, collapsing only toward forms attested in the same set."""
    # lemmatized surface form of each entity
    lemma_form = {e: " ".join(_lemma_tokens(e)) for e in entities}
    forms = set(lemma_form.values())
    # which single head-nouns are attested as standalone concepts
    one_word = {f for f in forms if f and " " not in f}

    canon: Dict[str, str] = {}
    for e, lf in lemma_form.items():
        toks = lf.split()
        # 2: drop leading qualifier(s) if the bare tail head-noun is attested
        while len(toks) >= 2 and toks[-1] in one_word and \
                " ".join(toks) not in one_word:
            toks = toks[1:]
            if " ".join(toks) in one_word:
                break
        form = " ".join(toks) if toks else lf
        # 3: WordNet synonym collapse for single-word concepts
        if " " not in form and form:
            form = _syn_rep(form, one_word)
        canon[e] = form or lf
    return canon


def apply_map(edges: List[Dict], canon: Dict[str, str]) -> List[Dict]:
    """Rewrite an edge list through the canonical map, dropping self-loops created
    by collapse and merging duplicate edges (keeping the max confidence)."""
    merged: Dict[tuple, Dict] = {}
    for e in edges:
        a, b = canon.get(e["a"], e["a"]), canon.get(e["b"], e["b"])
        if not a or not b or a == b:
            continue
        key = (a, b)
        prev = merged.get(key)
        ne = dict(e, a=a, b=b)
        if prev is None or ne.get("conf", 0) > prev.get("conf", 0):
            merged[key] = ne
    return list(merged.values())


def canonicalize(edges: List[Dict]) -> List[Dict]:
    """One call: build the map from the edges' entities and apply it."""
    ents = sorted({e["a"] for e in edges} | {e["b"] for e in edges})
    return apply_map(edges, build_map(ents))
