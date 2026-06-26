"""
normalize.py — deterministic entity canonicalization. No LLM, no embeddings.

The extractor keeps entities VERBATIM (the evidence must survive). But for the
GRAPH, "chronic stress" / "chronic psychological stress" / "stress" are one node,
and collapsing them is the single strongest measured lever in the stack: on the
151-paper gap_corpus it took 2-hop reasoning chains from 293 to 6,676 (~23×) and
cross-paper bridge nodes from 750 to 2,648 (+253%), at a 6% edge cost — because
edges that never met at a raw node now meet at a canonical one (the PCR effect:
connections become visible that lived in no single paper).

Two operations, both deterministic and inspectable:

  canonical(entity) -> the node key. lowercase, strip punctuation, drop a fixed
      stoplist of modifiers/qualifiers, then HEAD-NOUN collapse (keep the last
      two content tokens — English noun phrases head right: "chronic stress" and
      "stress" → "stress"). This is what makes variants meet.

  is_entity(canonical) -> False for sentence debris. The naive collapse also
      merges extraction artifacts ("et al", "e g", "we can", "does not") into
      fat fake hubs that poison the chains. Measured: filtering these dropped the
      naive 18,364 chains to 6,676 REAL ones while keeping 94% of edges. A node
      whose canonical form is pure function words / citation glue is not a thing
      in the world — it is dropped.

Used by build/build_causal_from_db.py to map raw triggers/outcomes to canonical
node keys before edges are written. The raw text stays in the DB; only the GRAPH
collapses.
"""
from __future__ import annotations

import re
from typing import Optional

# modifiers/qualifiers that do not change WHICH thing an entity is — dropping
# them is what makes "elevated cortisol" and "cortisol" the same node. These are
# severity/degree/direction words and bare determiners, not content nouns.
_STOP = frozenset("""
the a an of to in and or for with that this these those is are was were be been
by on as at from into chronic acute severe significant significantly increased
decreased high low elevated reduced patients with within during sustained
prolonged repeated frequent moderate mild marked overall further greater lesser
relative higher lower more less very largely mainly partly primarily mostly
""".split())

# canonical forms that are NOT entities — sentence debris the extractor picked
# up: citation glue, hedges, sentence-frame fragments. A node that collapses to
# one of these (or to pure function words) is dropped before it can become a
# fake hub. Measured: removes the "et al"(214) / "e g"(133) / "we can"(67) hubs.
_JUNK = frozenset([
    "et al", "e g", "i e", "et", "al", "we can", "we have", "we show",
    "does not", "is not", "are not", "has been", "have been", "can used",
    "can also", "can be", "such as", "this is", "it is", "they are",
    "we", "can", "used", "does", "this", "that", "such", "also",
])

# tokens that are never a content head on their own
_FUNC = frozenset("""
we i it he she they this that those these can does has have had is are was were
be been will would should could may might must not also such et al eg ie the a
an of to in on at by for with as from into and or but so then thus therefore
""".split())


def canonical(entity: str) -> str:
    """The graph node key for a raw entity string. Deterministic: lowercase,
    strip non-alphanumerics, drop the stoplist, head-noun collapse (last two
    content tokens). '' if nothing survives."""
    e = entity.lower().strip()
    e = re.sub(r"[^a-z0-9 -]", " ", e)
    toks = [t for t in e.split() if t and t not in _STOP]
    if len(toks) > 2:
        toks = toks[-2:]          # English NPs head right; keep the head
    return " ".join(toks)


def is_entity(canon: str) -> bool:
    """True if a canonical form names a thing in the world, False for debris.
    Used to drop fake hubs ('we can', 'et al') before they pollute the graph."""
    if not canon or len(canon) < 3:
        return False
    if canon in _JUNK:
        return False
    toks = canon.split()
    if all(t in _FUNC for t in toks):     # pure function words → not an entity
        return False
    if len(toks) == 1 and len(canon) < 4:
        return False
    return True


def normalize(entity: str) -> Optional[str]:
    """Convenience: canonical form if it is a real entity, else None (drop the
    edge). build/ uses this directly per endpoint."""
    c = canonical(entity)
    return c if is_entity(c) else None
