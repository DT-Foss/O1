"""
semantic_extractor.py — typed NON-causal relations for natural speech.

The causal extractor finds trigger→mechanism→outcome, perfect for reasoning but
most natural language is not causal. This extractor finds the relations that
make a brain able to *describe* things, not just explain cause and effect:

  is-a       taxonomy      "a candle is a light source"
  has-a      part / whole  "a candle has a wick"
  property   attribute     "wax is flammable"
  defines    definition    "combustion is rapid oxidation"
  does       agent-action  "the flame melts the wax"

It emits the SAME triplet shape (trigger, mechanism, outcome) — the relation
TYPE rides in the mechanism slot (prefixed "is-a", "has-a", ...). So these go
through the same DB → build → .causal path and mount as an ordinary module; the
brain just gains new ways to answer ("what is X", "describe X", "what does X
have"). Deterministic, no LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

# Each pattern captures (subject, object); the relation type is fixed per row.
# Patterns are anchored on copulas/possessives that are unambiguous in English.
REL_PATTERNS = [
    # definition: "X is defined as Y" / "X, defined as Y"
    (re.compile(r"^(.+?)\s+is\s+defined\s+as\s+(.+)$", re.I), "defines"),
    (re.compile(r"^(.+?)\s+refers?\s+to\s+(.+)$", re.I), "defines"),
    (re.compile(r"^(.+?)\s+means?\s+(.+)$", re.I), "defines"),
    # is-a: "an X is a Y" / "X is a kind of Y"
    (re.compile(r"^(.+?)\s+is\s+a\s+kind\s+of\s+(.+)$", re.I), "is-a"),
    (re.compile(r"^(.+?)\s+is\s+a\s+type\s+of\s+(.+)$", re.I), "is-a"),
    (re.compile(r"^(.+?)\s+(?:is|are)\s+an?\s+(.+)$", re.I), "is-a"),
    # has-a / part-of
    (re.compile(r"^(.+?)\s+(?:has|have|contains?|comprises?|includes?)\s+(.+)$", re.I), "has-a"),
    (re.compile(r"^(.+?)\s+consists?\s+of\s+(.+)$", re.I), "has-a"),
    (re.compile(r"^(.+?)\s+is\s+(?:part|a\s+part)\s+of\s+(.+)$", re.I), "part-of"),
    # property: "X is <single adjective>" / "X is very <adj>"
    (re.compile(r"^(.+?)\s+(?:is|are)\s+((?:very|highly|quite|extremely)\s+)?(\w+)$", re.I), "property"),
    # bare nominal predicate "X is Y Z" (no article) -> treat as is-a/definition
    (re.compile(r"^(.+?)\s+(?:is|are)\s+(.+)$", re.I), "is-a"),
]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_LEAD = re.compile(r"^(?:the|a|an|some|any|these|those|this|that)\s+", re.I)
_MAX = 70

# verbs that signal an AGENT-ACTION-OBJECT clause, not a copula relation
_ACTION = re.compile(
    r"^(.+?)\s+(melts?|burns?|heats?|cools?|moves?|holds?|carries?|"
    r"emits?|absorbs?|reflects?|forms?|breaks?|joins?|covers?|fills?|"
    r"surrounds?|drives?|lifts?|pushes?|pulls?)\s+(.+)$", re.I)


@dataclass
class SemTriplet:
    subject: str
    relation: str          # is-a / has-a / property / defines / part-of / does
    obj: str
    evidence: str
    source: str = ""
    domain: str = ""

    def as_dict(self) -> dict:
        # map onto the causal triplet schema: relation type in the mechanism
        return {
            "trigger": self.subject,
            "mechanism": self.relation,
            "outcome": self.obj,
            "confidence": "high",
            "evidence_sentence": self.evidence,
            "quantification": None,
            "domain": self.domain,
            "source": self.source,
        }


def _clean(s: str) -> str:
    s = s.strip().strip(" .,;:—-()[]")
    s = _LEAD.sub("", s)
    return s.strip()


def extract_from_sentence(sentence: str, domain: str = "",
                          source: str = "") -> List[SemTriplet]:
    s = sentence.strip().rstrip(".!?")
    if not (6 <= len(s) <= 200):
        return []
    # agent-action-object first (more specific than a copula)
    m = _ACTION.match(s)
    if m:
        subj, verb, obj = _clean(m.group(1)), m.group(2).lower(), _clean(m.group(3))
        if subj and obj and len(subj) <= _MAX and len(obj) <= _MAX:
            return [SemTriplet(subj, f"does:{verb}", obj, sentence.strip(),
                               source, domain)]
    for rx, rel in REL_PATTERNS:
        m = rx.match(s)
        if not m:
            continue
        subj = _clean(m.group(1))
        # object is the LAST captured group (property pattern has an optional
        # intensifier group before the adjective)
        obj = _clean("".join(g for g in m.groups()[1:] if g))
        if not subj or not obj or subj.lower() == obj.lower():
            continue
        if len(subj) > _MAX or len(obj) > _MAX:
            continue
        return [SemTriplet(subj, rel, obj, sentence.strip(), source, domain)]
    return []


def extract_from_text(text: str, domain: str = "",
                      source: str = "") -> List[SemTriplet]:
    out: List[SemTriplet] = []
    for sent in _SENT_SPLIT.split(text):
        out.extend(extract_from_sentence(sent.strip(), domain, source))
    return out
