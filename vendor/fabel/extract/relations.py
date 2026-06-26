"""
relations.py — richer connection types beyond the plain binary triplet.

A triplet (A, mechanism, B) is a binary directed edge — the simplest unit, but
some real relations LIE when forced into a pair. These functions extract the
extra structure deterministically (regex on measured language signals) and
return it as fields to attach to a triplet. Each is independent and optional:
detected → fill the field; not detected → return None, triplet stays plain.

The five types (from the design discussion):
  1. n-ary / population   "...in adults over 40", "...among smokers"
  2. conditional          "X causes Y when/if Z"  (the edge is conditioned)
  3. hyperedge co-causes  "A and B together cause Y", "combined, A,B,C ..."
  4. effect size          odds ratio / hazard ratio / relative risk (a real
                          measure, richer than high/medium/low confidence)
  5. temporal ordering    "before/after/then" — sequence, not causation
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple


# ---------------------------------------------------------------- 1. population

_POPULATION = re.compile(
    r"\b(?:in|among|for|amongst)\s+("
    r"(?:the\s+)?(?:adults?|children|patients?|men|women|smokers?|people|"
    r"individuals?|elderly|infants?|adolescents?|"
    r"\w+\s+(?:patients?|adults?|individuals?|groups?))"
    r"(?:\s+(?:over|under|aged|with|who|above|below)\s+[\w\s.]+?)?)"
    r"(?=[,.;]|$|\s+(?:and|but|when|if))", re.I)


def population(sentence: str) -> Optional[str]:
    """An n-ary qualifier scoping who/where the relation holds."""
    m = _POPULATION.search(sentence)
    if m:
        p = m.group(1).strip(" ,.")
        # avoid grabbing the whole rest of the sentence
        return p if len(p.split()) <= 7 else None
    return None


# ---------------------------------------------------------------- 2. conditional

_CONDITION = re.compile(
    r"[,]?\s*\b(?:but\s+)?(?:only\s+)?(?:when|if|unless|provided\s+that|"
    r"as\s+long\s+as|in\s+the\s+presence\s+of|in\s+the\s+absence\s+of|"
    r"given|whenever)\s+(.+?)(?=[,.;]|$)", re.I)


def condition(sentence: str) -> Optional[str]:
    """The condition gating a causal edge: 'X causes Y WHEN Z' -> Z.
    This is the field that turns a LYING triplet into an honest one — without
    it, 'X causes Y' is asserted unconditionally when the text said otherwise."""
    m = _CONDITION.search(sentence)
    if m:
        cond = m.group(1).strip(" ,.")
        # a condition should be a clause, not the whole sentence tail
        if 1 <= len(cond.split()) <= 12 and cond.lower() not in (
                "possible", "necessary", "needed"):
            return cond
    return None


# ---------------------------------------------------------------- 3. hyperedge

_JOINT = re.compile(
    r"\b(?:together|combined|jointly|in\s+combination|"
    r"simultaneously|collectively|in\s+concert)\b|^combined,", re.I)
# "A and B and C <verb>" as a compound subject of a causal verb
_COMPOUND_SUBJ = re.compile(
    r"^\s*(.+?\b(?:and|,)\s+.+?)\s+"
    r"(?:cause|causes|produce|produces|trigger|triggers|lead|leads\s+to|"
    r"result\s+in|results\s+in|create|creates|drive|drives)\b", re.I)


def co_causes(sentence: str, trigger: str) -> Optional[List[str]]:
    """If the cause is a CONJUNCTION acting jointly ('heat and drought and wind
    cause fire'), return the list of co-causes. A plain 'A and B' subject only
    counts as a hyperedge when a joint marker is present OR the trigger itself
    is a conjunction — otherwise A and B might cause Y independently."""
    has_joint = bool(_JOINT.search(sentence))
    # strip a trailing joint marker that leaked into the last conjunct
    trig = _JOINT.sub("", trigger).strip()
    parts = [p.strip() for p in re.split(r"\s*,\s*|\s+and\s+", trig)
             if p.strip()]
    if len(parts) >= 2 and (has_joint or len(parts) >= 3):
        return parts
    if has_joint:
        m = _COMPOUND_SUBJ.match(sentence)
        if m:
            sub = [p.strip() for p in re.split(r"\s*,\s*|\s+and\s+", m.group(1))
                   if p.strip()]
            if len(sub) >= 2:
                return sub
    return None


# ---------------------------------------------------------------- 4. effect size

_EFFECT_SIZE = re.compile(
    r"\b("
    r"(?:odds\s+ratio|hazard\s+ratio|relative\s+risk|risk\s+ratio|"
    r"(?:OR|HR|RR))\s*(?:of|=|:)?\s*\d+(?:\.\d+)?"
    r"(?:\s*\(?95%\s*CI[\s:]*[\d.\s,–-]+\)?)?"
    r"|\d+(?:\.\d+)?\s*-?\s*fold\s+(?:increase|decrease|risk|higher|lower)"
    r"|\d+(?:\.\d+)?[xX]\s+(?:more|less|higher|lower|likely)"
    r")", re.I)


def effect_size(sentence: str) -> Optional[str]:
    """A real effect measure (OR/HR/RR/fold) — richer than scalar confidence."""
    m = _EFFECT_SIZE.search(sentence)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------- 5. temporal

_TEMPORAL = re.compile(
    r"\b(before|after|prior\s+to|following|subsequently|then|"
    r"once|first|later|earlier|preceding|simultaneously)\b", re.I)
# pure-sequence verbs that signal ordering rather than causation
_SEQUENCE = re.compile(
    r"\b(precedes?|follows?|precede|comes?\s+before|comes?\s+after)\b", re.I)


def temporal(sentence: str) -> Optional[str]:
    """A temporal ordering marker — flags that the link is sequence, which may
    or may not be causal ('A then B' is order, not necessarily cause)."""
    m = _SEQUENCE.search(sentence) or _TEMPORAL.search(sentence)
    return m.group(1).lower().strip() if m else None


# ---------------------------------------------------------- attach-all helper

def enrich(triplet, sentence: str):
    """Run all five detectors and attach what fires to a RawTriplet in place.
    Returns the triplet for chaining."""
    triplet.population = population(sentence)
    triplet.condition = condition(sentence)
    triplet.co_causes = co_causes(sentence, triplet.trigger)
    triplet.effect_size = effect_size(sentence)
    triplet.temporal = temporal(sentence)
    return triplet
