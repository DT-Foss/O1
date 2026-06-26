"""
rule_extractor.py — deterministic causal triplet extraction. No LLM, no MLX.

Replaces the MLX few-shot extractor (extract_causal_triplets_v2.py) with a
pure rule engine: causal connectives are MEASURED structure in the text, so a
finite, inspectable pattern set finds them — same philosophy as the rest of
the stack (measure structure, never guess).

For each sentence:
  1. split into clauses around a causal CONNECTIVE (causes / leads to /
     reduces / because / due to / ...)
  2. the connective itself becomes the MECHANISM (its polarity is known)
  3. the clause before/after becomes TRIGGER / OUTCOME (order depends on
     whether the connective is forward "A causes B" or backward "B due to A")
  4. attach a verbatim quantification if a number-pattern hits the sentence
  5. confidence is the connective's strength (explicit verb > soft hedge)

Output triplet dict matches the pipeline schema exactly:
  trigger, mechanism, outcome, confidence ('high'/'medium'/'low'),
  evidence_sentence, quantification (or None), domain, source

Each triplet then passes the existing 14-step Foss gate (validate_triplet_v2)
before it is kept — identical quality bar to the LLM path.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from relations import enrich as _enrich


# --- causal connectives -----------------------------------------------------
# Each entry: (regex, direction, confidence, mechanism_normal_form).
# direction 'fwd'  : "TRIGGER <conn> OUTCOME"   (A causes B)
# direction 'bwd'  : "OUTCOME <conn> TRIGGER"   (B because of A)
# The regex matches the connective only; clause text is taken around the span.

_FWD = "fwd"
_BWD = "bwd"

CONNECTIVES: List[Tuple[str, str, str, str]] = [
    # strong, explicit, forward
    (r"\bcauses?\b", _FWD, "high", "causes"),
    (r"\bcausing\b", _FWD, "high", "causes"),
    (r"\bleads?\s+to\b", _FWD, "high", "leads to"),
    (r"\bleading\s+to\b", _FWD, "high", "leads to"),
    (r"\bresults?\s+in\b", _FWD, "high", "results in"),
    (r"\bresulting\s+in\b", _FWD, "high", "results in"),
    (r"\btriggers?\b", _FWD, "high", "triggers"),
    (r"\binduces?\b", _FWD, "high", "induces"),
    (r"\bproduces?\b", _FWD, "high", "produces"),
    (r"\bgenerates?\b", _FWD, "high", "generates"),
    # directional effect verbs (polarity matters downstream)
    (r"\bincreases?\b", _FWD, "high", "increases"),
    (r"\benhances?\b", _FWD, "high", "enhances"),
    (r"\bimproves?\b", _FWD, "high", "improves"),
    (r"\breduces?\b", _FWD, "high", "reduces"),
    (r"\bdecreases?\b", _FWD, "high", "decreases"),
    (r"\binhibits?\b", _FWD, "high", "inhibits"),
    (r"\bprevents?\b", _FWD, "high", "prevents"),
    (r"\bsuppress(?:es)?\b", _FWD, "high", "suppresses"),
    (r"\brelieves?\b", _FWD, "high", "relieves"),
    (r"\balleviates?\b", _FWD, "high", "alleviates"),
    (r"\bmitigates?\b", _FWD, "high", "mitigates"),
    (r"\bworsens?\b", _FWD, "high", "worsens"),
    (r"\baccelerates?\b", _FWD, "high", "accelerates"),
    (r"\bslows?\b", _FWD, "high", "slows"),
    (r"\bdamages?\b", _FWD, "high", "damages"),
    (r"\bimpairs?\b", _FWD, "high", "impairs"),
    (r"\bblocks?\b", _FWD, "high", "blocks"),
    (r"\benables?\b", _FWD, "high", "enables"),
    # disruption / change verbs (incl. past tense, common in prose)
    (r"\bdisrupt(?:s|ed)?\b", _FWD, "high", "disrupts"),
    (r"\bboosts?\b", _FWD, "high", "boosts"),
    (r"\bweakens?\b", _FWD, "high", "weakens"),
    (r"\bstrengthens?\b", _FWD, "high", "strengthens"),
    (r"\bdisabl(?:es?|ed)\b", _FWD, "high", "disables"),
    (r"\brestricts?\b", _FWD, "high", "restricts"),
    (r"\bstimulates?\b", _FWD, "high", "stimulates"),
    (r"\bdiminishes?\b", _FWD, "high", "diminishes"),
    (r"\berodes?\b", _FWD, "high", "erodes"),
    (r"\bdepletes?\b", _FWD, "high", "depletes"),
    (r"\bkills?\b", _FWD, "high", "kills"),
    (r"\bdestroys?\b", _FWD, "high", "destroys"),
    (r"\braises?\b", _FWD, "high", "raises"),
    (r"\blowers?\b", _FWD, "high", "lowers"),
    (r"\bspeeds?\s+up\b", _FWD, "high", "speeds up"),
    (r"\bfuels?\b", _FWD, "high", "fuels"),
    (r"\bpromotes?\b", _FWD, "high", "promotes"),
    (r"\bdrives?\b", _FWD, "high", "drives"),
    (r"\bcontributes?\s+to\b", _FWD, "medium", "contributes to"),
    # backward (effect first)
    (r"\bbecause\s+of\b", _BWD, "high", "is caused by"),
    (r"\bdue\s+to\b", _BWD, "high", "is caused by"),
    (r"\bcaused\s+by\b", _BWD, "high", "is caused by"),
    (r"\bresulting\s+from\b", _BWD, "high", "results from"),
    (r"\bowing\s+to\b", _BWD, "medium", "is caused by"),
    (r"\battributed\s+to\b", _BWD, "medium", "is attributed to"),
    # soft / hedged (lower confidence)
    (r"\bassociated\s+with\b", _FWD, "low", "is associated with"),
    (r"\blinked\s+to\b", _FWD, "low", "is linked to"),
    (r"\bcorrelates?\s+with\b", _FWD, "low", "correlates with"),
    (r"\bmay\s+lead\s+to\b", _FWD, "low", "may lead to"),
    (r"\bcan\s+cause\b", _FWD, "medium", "can cause"),
]

_COMPILED = [(re.compile(rx, re.I), d, c, m) for rx, d, c, m in CONNECTIVES]

# --- WordNet-harvested causal verb lexicon (built once, loaded as static JSON;
# no model/NLTK at runtime). 1100+ verbs with polarity, covering the long tail
# the hand-written CONNECTIVES miss (threaten, erode, displace, ...). The hand
# patterns above still win first (they carry exact mechanism wording); the
# lexicon is the fallback for any verb form not explicitly listed.
import json as _json
_LEXICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "causal_verbs.json")
try:
    _CAUSAL_VERBS = _json.load(open(_LEXICON_PATH, encoding="utf-8"))
except Exception:
    _CAUSAL_VERBS = {}

# verbs already covered by an explicit CONNECTIVE pattern — don't double-list
_EXPLICIT_VERBS = frozenset(
    m.split()[0] for *_, m in CONNECTIVES if m.split())


def _lemmatize_verb(word: str) -> str:
    """Crude deterministic de-inflection: 3rd-person -s and past -ed/-ied.
    No model — just the regular-verb suffix rules, enough to hit the lexicon."""
    w = word.lower()
    cands = [
        w,
        re.sub(r"ies$", "y", w),
        re.sub(r"(ss|sh|ch|x|z)es$", r"\1", w),
        re.sub(r"es$", "", w),
        re.sub(r"s$", "", w),
        re.sub(r"ied$", "y", w),
        re.sub(r"ed$", "", w),
        re.sub(r"ed$", "e", w),
        re.sub(r"ing$", "", w),
        re.sub(r"ing$", "e", w),
    ]
    # doubled final consonant before -ed/-ing ("spurred"->"spur", "stopped"->"stop")
    m = re.match(r"^(.+?)([bcdfghjklmnpqrstvwz])\2(?:ed|ing)$", w)
    if m:
        cands.append(m.group(1) + m.group(2))
    for cand in cands:
        if cand in _CAUSAL_VERBS:
            return cand
    return ""


_VERB_TOKEN = re.compile(r"\b([a-z]+(?:s|ed|es|ies|ied|ing)?)\b", re.I)
# a causal verb followed directly by a preposition or an -ly adverb is
# intransitive here ("opens AT nine", "burned BRIGHTLY") — not SUBJ-VERB-OBJ.
_INTRANS_AFTER = re.compile(
    r"^\s+(?:at|in|on|to|from|into|onto|up|down|out|off|over|under|away|"
    r"back|along|around|through|by|with|for|of|near|toward|"
    r"\w+ly\b)", re.I)


# optional spaCy POS tagger: disambiguates which token is actually the VERB
# ("levels" is a noun in "sea levels threaten X", not the verb). Lazy-loaded;
# if absent, the scanner falls back to surface heuristics. This is the only
# place a model is touched, and only to PICK among lexicon verbs — the causal
# knowledge still comes from WordNet, deterministically.
_NLP = None
_NLP_TRIED = False


def _verb_positions(sentence: str):
    """Token spans tagged VERB by spaCy, or None if spaCy is unavailable."""
    global _NLP, _NLP_TRIED
    if not _NLP_TRIED:
        _NLP_TRIED = True
        try:
            import spacy
            _NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        except Exception:
            _NLP = None
    if _NLP is None:
        return None
    doc = _NLP(sentence)
    return {(t.idx, t.idx + len(t.text)) for t in doc if t.pos_ == "VERB"}


def _lexicon_scan(sentence: str):
    """Find a lexicon causal verb used TRANSITIVELY. With spaCy, only tokens
    actually tagged VERB are considered (so 'sea levels threaten X' picks
    'threaten', not the noun 'levels'). The transitivity guard keeps
    non-causal verbs ('opens at nine') from firing."""
    verb_spans = _verb_positions(sentence)   # None if spaCy missing
    for m in _VERB_TOKEN.finditer(sentence):
        if verb_spans is not None and (m.start(), m.end()) not in verb_spans:
            continue                          # POS says this token isn't a verb
        lemma = _lemmatize_verb(m.group(1))
        if not lemma or lemma in _EXPLICIT_VERBS:
            continue
        rest = sentence[m.end():]
        if _INTRANS_AFTER.match(rest):
            continue
        if not _clean(rest, take_last=False):
            continue
        return (m.start(), m.end(), m.group(1).lower(), _CAUSAL_VERBS[lemma])
    return None


# quantification patterns (verbatim numbers) — subset of the pipeline's set
QUANT_PATTERNS = [
    r"\d+(?:\.\d+)?\s*%", r"\d+(?:\.\d+)?\s*percent",
    r"\d+(?:\.\d+)?[xX]\b", r"\d+(?:\.\d+)?\s*(?:fold|times)",
    r"\$\s*\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|M|B|K)?",
    r"\d+(?:\.\d+)?\s*(?:mg|kg|ml|mmol|mol|nm|mm|cm|km|ms|Hz|kHz|MHz|GHz)",
    r"\d+(?:\.\d+)?\s*(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)",
    r"\d+(?:,\d{3})*\s+(?:patients?|subjects?|samples?|cases?)",
    r"\bp\s*[<>=]\s*0?\.\d+", r"\d+(?:\.\d+)?\s*(?:±|\+/-)\s*\d+(?:\.\d+)?",
]
_QUANT = [re.compile(p, re.I) for p in QUANT_PATTERNS]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
# trim leading/trailing function-word noise from a clause
_CLAUSE_TRIM = re.compile(
    r"^(?:the|a|an|that|which|this|these|those|and|but|so|when|while|"
    r"if|as|then|thus|therefore|however|moreover)\s+", re.I)


@dataclass
class RawTriplet:
    trigger: str
    mechanism: str
    outcome: str
    confidence: str
    evidence_sentence: str
    quantification: Optional[str] = None
    domain: str = ""
    source: str = ""
    # --- richer connection types (all optional; a plain triplet leaves them
    # empty, so nothing downstream breaks) ---
    condition: Optional[str] = None      # (2) "X causes Y | when Z"
    co_causes: Optional[list] = None     # (3) hyperedge: {A,B,C} jointly -> Y
    effect_size: Optional[str] = None    # (4) odds ratio / hazard ratio / etc.
    population: Optional[str] = None      # (1) n-ary: "in adults over 40"
    temporal: Optional[str] = None       # (5) "before"/"after"/"then" ordering

    def as_dict(self) -> Dict:
        d = {
            "trigger": self.trigger,
            "mechanism": self.mechanism,
            "outcome": self.outcome,
            "confidence": self.confidence,
            "evidence_sentence": self.evidence_sentence,
            "quantification": self.quantification,
            "domain": self.domain,
            "source": self.source,
        }
        # only attach the richer fields when present, so plain triplets stay plain
        if self.condition:
            d["condition"] = self.condition
        if self.co_causes:
            d["co_causes"] = self.co_causes
        if self.effect_size:
            d["effect_size"] = self.effect_size
        if self.population:
            d["population"] = self.population
        if self.temporal:
            d["temporal"] = self.temporal
        return d


_DANGLE = re.compile(
    r"\s+(?:is|was|are|were|be|been|largely|mainly|partly|primarily|mostly|"
    r"this|that|it|which)$", re.I)


# a clause that reduces to only these is segmentation debris, not an entity
_NON_ENTITY = frozenset(
    "is was are were be been this that it which the a an and but so".split())

# boundaries that end a noun phrase — when an extracted clause is too long, the
# entity is the phrase NEAREST the connective, cut at the closest such marker.
_NP_BOUNDARY = re.compile(
    r"\s*(?:[;,]|—|\b(?:and|but|or|so|because|which|that|when|while|if|as|"
    r"then|thus|therefore|however|although|though|since|after|before|"
    r"with|for|to)\b)\s*", re.I)

# markers that END an outcome/trigger noun phrase: a trailing adjunct
# ("inflammation BY inhibiting...", "breathlessness, WHICH...") is not part of
# the entity. Cut the entity at the first of these, always (not only when long).
# Also cut population ("in adults over 40") and effect-size ("with a hazard
# ratio of 2.5") adjuncts — they are captured as their own fields, so they must
# not pollute the outcome.
_NP_TAIL = re.compile(
    r"\s+(?:by|via|through|which|whilst|while|that|when|where|because|if|"
    r"due\s+to|so\s+that|in\s+order\s+to|thereby|thus|leading\s+to|"
    r"resulting\s+in|with\s+a[n]?\s+(?:odds|hazard|risk|relative)|"
    r"in\s+(?:adults?|children|patients?|men|women|smokers?|people)|"
    r"among|amongst)\b.*$", re.I)
_NP_TAIL_PUNCT = re.compile(r"\s*[,;—].*$")   # ", which in turn ..."

_MAX_ENTITY_CHARS = 50


def _head_np(clause: str) -> str:
    """Keep the leading noun phrase: drop a trailing adjunct/relative clause.
    'inflammation by inhibiting COX-2' -> 'inflammation';
    'breathlessness, which in turn ...' -> 'breathlessness'."""
    clause = _NP_TAIL.sub("", clause)
    clause = _NP_TAIL_PUNCT.sub("", clause)
    return clause.strip()


def _spacy_np(clause: str, take_last: bool):
    """Pick the head noun phrase of an over-long clause via spaCy noun_chunks
    (linguistically correct, vs. character truncation). Returns None if spaCy is
    absent or finds nothing useful. take_last → the chunk nearest the connective
    (end of a left clause); else the first chunk (start of a right clause)."""
    _verb_positions("")  # triggers lazy spaCy load
    if _NLP is None:
        return None
    chunks = [c.text.strip() for c in _NLP(clause).noun_chunks
              if len(c.text.strip()) >= 4]
    if not chunks:
        return None
    return chunks[-1] if take_last else chunks[0]


def _trim_to_np(clause: str, take_last: bool) -> str:
    """Cut an over-long clause down to the noun phrase nearest the connective."""
    if len(clause) <= _MAX_ENTITY_CHARS:
        return clause
    # prefer a real noun-phrase chunk (spaCy) over character/boundary cutting
    np = _spacy_np(clause, take_last)
    if np and len(np) <= _MAX_ENTITY_CHARS:
        return np
    pieces = [p.strip() for p in _NP_BOUNDARY.split(clause) if p.strip()]
    if not pieces:
        return clause[:_MAX_ENTITY_CHARS].rsplit(" ", 1)[0]
    seq = pieces[::-1] if take_last else pieces
    acc: list = []
    for p in seq:
        acc.append(p)
        if len(" ".join(acc)) >= 15:
            break
    return " ".join(acc[::-1] if take_last else acc)


# discourse / section-marker words that are never a standalone causal concept.
# A cross-paper hypothesis must not bridge on these: measured on the gap corpus,
# "result" ALONE carried 23% of all cross-paper chains, and the discourse top-12
# carried 49% — pure noise masquerading as connections.
_DISCOURSE_NP = frozenset(
    "result results analysis observation observations background discussion "
    "introduction conclusion conclusions author authors method methods approach "
    "data value values number numbers case cases study studies trial trials "
    "review reviews work paper section figure table example model time point "
    "points way effect effects change increase decrease evidence outcome "
    "outcomes finding findings setting settings use using".split())

# Opt-in: reduce every entity to its concept head noun phrase, not only the
# over-long ones. Off by default so the tuned gold F1 is unchanged; the
# cross-paper hypothesis pipeline turns it on, where concept-level entities are
# what make chains real. Measured lift: 76% -> 96% concept-clean entities, and
# 0 -> 20 real cross-paper hypotheses on a 15-review medical sample.
_CONCEPT_MODE = False


def set_concept_mode(on: bool = True) -> None:
    """Toggle concept head-NP reduction of entities (see _concept_np)."""
    global _CONCEPT_MODE
    _CONCEPT_MODE = bool(on)


def _concept_np(clause: str, take_last: bool) -> str:
    """Reduce a clause to its core concept: the content tokens (NOUN/PROPN/ADJ,
    non-stop) of its connective-nearest noun chunk. 'other two trials assigned
    women' -> 'women'; 'one method that has been evaluated' -> 'method'. Returns
    '' when the concept is purely discourse, so the triplet is dropped rather
    than bridged on noise."""
    _verb_positions("")  # lazy spaCy load
    if _NLP is None or not clause:
        return clause
    doc = _NLP(clause)
    chunks = list(doc.noun_chunks)
    if chunks:
        c = chunks[-1] if take_last else chunks[0]
        toks = [t.text.lower() for t in c
                if t.pos_ in ("NOUN", "PROPN", "ADJ") and not t.is_stop]
    else:
        toks = [t.text.lower() for t in doc if t.pos_ in ("NOUN", "PROPN")]
        if not toks:
            return clause
        toks = [toks[-1]] if take_last else [toks[0]]
    # drop discourse tokens INSIDE the concept ("abstract background betamimetics"
    # -> "betamimetics"); if nothing real survives, it was pure discourse
    toks = [w for w in toks if w not in _DISCOURSE_NP]
    return " ".join(toks)


def _clean(clause: str, take_last: bool = False) -> str:
    clause = clause.strip().strip(" .,;:—-()[]")
    clause = _CLAUSE_TRIM.sub("", clause)
    prev = None
    while prev != clause:
        prev = clause
        clause = _DANGLE.sub("", clause).strip()
    if not clause or all(w.lower() in _NON_ENTITY for w in clause.split()):
        return ""
    # the OUTCOME (head phrase) drops trailing adjuncts always; the TRIGGER
    # (tail phrase) keeps its head but still gets length-capped
    if not take_last:
        clause = _head_np(clause)
    clause = _trim_to_np(clause, take_last)
    if _CONCEPT_MODE:
        clause = _concept_np(clause, take_last)
    return clause.strip()


def _find_quant(sentence: str) -> Optional[str]:
    for rx in _QUANT:
        m = rx.search(sentence)
        if m:
            return m.group(0).strip()
    return None


# unambiguous causal markers — these are ALWAYS verbs/connectives, never
# nouns, so they win over effect verbs ("damages", "reduces") that double as
# nouns ("the lung damage causes X": prefer 'causes', not the noun 'damage').
_UNAMBIGUOUS = frozenset({
    "causes", "leads to", "results in", "triggers", "induces",
    "is caused by", "results from", "can cause"})


# sentence-initial "Because X, Y" / "Since X, Y": X is the cause of Y. The
# clauses are split by the comma, not a verb, so handle it before the verb scan.
_BECAUSE_LEAD = re.compile(r"^\s*(?:because|since|as)\s+(.+?),\s+(.+)$", re.I)

# "A <happens> before/after B <happens>" — temporal sequence, not causation.
# Captures the two clauses around the ordering word.
_TEMPORAL_EDGE = re.compile(
    r"^(?P<a>.+?)\s+(?P<rel>before|after|prior\s+to|following|preceding)\s+"
    r"(?P<b>.+)$", re.I)


def _has_verb(clause: str) -> bool:
    """True if the clause contains a finite verb (a real event), so a temporal
    edge connects two events, not an event and a bare time word ('lunch')."""
    spans = _verb_positions(clause)
    if spans is not None:
        return len(spans) > 0
    # no spaCy: a -s/-ed/-ing token that isn't a common noun is a fair proxy
    return bool(re.search(r"\b\w+(?:s|ed|ing)\b", clause, re.I))


# a parenthetical aside set off by commas ("X, often due to stress, worsens Y")
# is not part of the main causal claim — strip it so the head verb wins.
_APPOSITIVE = re.compile(
    r",\s+(?:often|sometimes|usually|typically|largely|partly|mainly|common|"
    r"frequent|rare|due\s+to|owing\s+to|because\s+of|which\s+is|"
    r"a\s+kind\s+of)\b[^,]*,",
    re.I)


def extract_from_sentence(sentence: str, domain: str = "",
                          source: str = "") -> List[RawTriplet]:
    """All causal triplets a single sentence yields (often 0 or 1)."""
    original = sentence            # richer-relation detectors need the full text
    quant = _find_quant(sentence)
    # drop a comma-set-off appositive so the main clause's verb is the connective
    sentence = _APPOSITIVE.sub(" ", sentence)
    # "Because X, Y" — X causes Y (comma-split, no connecting verb)
    m = _BECAUSE_LEAD.match(sentence.strip())
    if m:
        cause = _clean(m.group(1), take_last=False)
        effect = _clean(m.group(2), take_last=False)
        if cause and effect and cause.lower() != effect.lower():
            return [_enrich(RawTriplet(trigger=cause, mechanism="is caused by",
                            outcome=effect, confidence="high",
                            evidence_sentence=sentence.strip(),
                            quantification=quant, domain=domain,
                            source=source), original)]
    # collect every connective hit, then pick the BEST one: unambiguous causal
    # markers first, then earliest position. This stops a noun that looks like
    # an effect verb ("damage") from beating the real verb ("causes").
    candidates = []
    for rx, direction, conf, mech in _COMPILED:
        m = rx.search(sentence)
        if not m:
            continue
        left = _clean(sentence[:m.start()], take_last=True)
        right = _clean(sentence[m.end():], take_last=False)
        if not left or not right:
            continue
        if direction == _FWD:
            trigger, outcome = left, right
        else:
            trigger, outcome = right, left
        priority = 0 if mech in _UNAMBIGUOUS else 1
        candidates.append(((priority, m.start()), RawTriplet(
            trigger=trigger, mechanism=mech, outcome=outcome,
            confidence=conf, evidence_sentence=sentence.strip(),
            quantification=quant, domain=domain, source=source)))
    if candidates:
        candidates.sort(key=lambda c: c[0])
        return [_enrich(candidates[0][1], original)]
    # FALLBACK: no explicit connective matched — scan for any WordNet causal
    # verb (the long tail). Same SUBJ-VERB-OBJ shape, verb form as mechanism.
    hit = _lexicon_scan(sentence)
    if hit:
        start, end, verb, _pol = hit
        left = _clean(sentence[:start], take_last=True)
        right = _clean(sentence[end:], take_last=False)
        if left and right and left.lower() != right.lower():
            return [_enrich(RawTriplet(trigger=left, mechanism=verb,
                            outcome=right, confidence="medium",
                            evidence_sentence=sentence.strip(),
                            quantification=quant, domain=domain,
                            source=source), original)]
    # TEMPORAL fallback: "A before/after/then B" is sequence, not causation.
    # Emit it honestly as a temporal edge (precedes/follows), low confidence —
    # it MAY be causal but the text only stated order. Guard: BOTH clauses must
    # contain a verb (two real EVENTS), else "meeting starts after lunch" — a
    # bare time reference — would wrongly become an edge.
    seq = _TEMPORAL_EDGE.match(sentence.strip())
    if seq and _has_verb(seq.group("a")) and _has_verb(seq.group("b")):
        a = _clean(seq.group("a"), take_last=True)
        rel_word = seq.group("rel").lower()
        b = _clean(seq.group("b"), take_last=False)
        if a and b and a.lower() != b.lower():
            mech = "precedes" if rel_word in ("before", "prior to",
                                              "preceding") else "follows"
            t = RawTriplet(trigger=a, mechanism=mech, outcome=b,
                           confidence="low", evidence_sentence=sentence.strip(),
                           quantification=quant, domain=domain, source=source)
            t.temporal = rel_word
            return [t]
    return []


# a causal sentence starting with a demonstrative/pronoun points BACK to the
# previous sentence — "X happens. This causes Y." The link breaks across the
# boundary unless we resolve "This" to the prior sentence's topic. Measured at
# ~22% of causal sentences in scientific prose. This is the deterministic form
# of "attention on the transition" — coreference, no transformer.
_COREF_LEAD = re.compile(
    r"^\s*(This|These|That|Those|It|They|Such)\b\s*(\w+)?", re.I)


def _resolve_coref(sentence: str, prev_topic: Optional[str]) -> str:
    """If the sentence opens with a back-pointing pronoun and we know the prior
    sentence's topic, substitute it so the causal link survives the boundary."""
    if not prev_topic:
        return sentence
    m = _COREF_LEAD.match(sentence)
    if not m:
        return sentence
    # replace just the leading pronoun (keep the rest, incl. a following noun)
    return prev_topic + sentence[m.end(1):]


def _topic_of(sentence: str) -> Optional[str]:
    """The topic an anaphor would refer to: the subject noun phrase (spaCy if
    present, else the first noun-ish chunk)."""
    _verb_positions("")
    if _NLP is not None:
        doc = _NLP(sentence)
        for chunk in doc.noun_chunks:
            return chunk.text.strip()
        return None
    # fallback: first few words up to a verb-ish token
    words = sentence.split()
    return " ".join(words[:4]) if words else None


def extract_from_text(text: str, domain: str = "",
                      source: str = "") -> List[RawTriplet]:
    """Extract causal triplets sentence by sentence, resolving back-pointing
    pronouns ("This causes Y") to the previous sentence's topic so cross-
    sentence causal links survive."""
    triplets: List[RawTriplet] = []
    prev_topic: Optional[str] = None
    for sent in _SENT_SPLIT.split(text):
        sent = sent.strip()
        if 20 <= len(sent) <= 600:
            resolved = _resolve_coref(sent, prev_topic)
            triplets.extend(extract_from_sentence(resolved, domain, source))
        # remember this sentence's topic for the next one's anaphora
        if len(sent) >= 10:
            prev_topic = _topic_of(sent) or prev_topic
    return triplets
