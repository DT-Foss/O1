"""
build_verb_lexicon.py — harvest causal verbs from WordNet, once, into a static
lexicon the rule extractor loads. WordNet is used only at BUILD time; at runtime
the extractor reads a plain JSON list — no model, no NLTK dependency, fully
deterministic and offline.

Method: seed with known causal verbs, then walk WordNet's verb hierarchy —
hyponyms (more specific kinds) of causal-change verbs, plus synonyms of the
seeds. Each verb is tagged with a polarity (increase / decrease / neutral) from
which seed cluster it descends, so the extractor keeps the +/- direction it
already uses. Produces extract/causal_verbs.json.
"""
from __future__ import annotations

import json
import os

import nltk
nltk.download("wordnet", quiet=True)
from nltk.corpus import wordnet as wn

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "causal_verbs.json")

# seed clusters: known causal verbs grouped by polarity. WordNet expands each.
SEEDS = {
    "increase": ["increase", "raise", "boost", "amplify", "intensify",
                 "accelerate", "enhance", "strengthen", "stimulate", "fuel"],
    "decrease": ["decrease", "reduce", "lower", "diminish", "suppress",
                 "weaken", "inhibit", "deplete", "impair", "slow", "erode",
                 "block", "prevent", "restrict", "dampen"],
    "cause":    ["cause", "produce", "trigger", "induce", "generate", "create",
                 "lead", "drive", "provoke", "spark", "yield", "bring"],
    "damage":   ["damage", "destroy", "harm", "disrupt", "impair", "degrade",
                 "ruin", "undermine", "kill", "threaten", "displace", "erode",
                 "paralyze", "pollute", "cripple", "hamper", "poison",
                 "corrode", "contaminate", "wreck", "devastate"],
    "improve":  ["improve", "enhance", "benefit", "boost", "promote", "enable",
                 "facilitate", "aid", "ease", "relieve", "alleviate", "heal",
                 "spur", "foster", "encourage", "accelerate", "advance"],
}


def expand(seed_word, depth=1):
    """Synonyms + hyponyms of a seed verb, lemma forms only."""
    out = set()
    for syn in wn.synsets(seed_word, pos="v"):
        for lemma in syn.lemmas():
            w = lemma.name().replace("_", " ")
            if " " not in w and w.isalpha():
                out.add(w.lower())
        # one level of hyponyms (more specific causal kinds)
        for hypo in syn.hyponyms():
            for lemma in hypo.lemmas():
                w = lemma.name().replace("_", " ")
                if " " not in w and w.isalpha():
                    out.add(w.lower())
    return out


def main():
    lexicon = {}     # verb -> polarity cluster
    for polarity, seeds in SEEDS.items():
        for seed in seeds:
            for verb in expand(seed):
                # first cluster wins; seeds themselves are authoritative
                lexicon.setdefault(verb, polarity)
            lexicon[seed] = polarity   # seed overrides any prior softer tag
    # drop over-general / intransitive / motion verbs WordNet's hyponym walk
    # drags in — these fire on non-causal sentences ("the library opens at
    # nine", "prices rise"). Causal extraction wants transitive change verbs.
    NOISE = (
        "be have do make get go come take give use see know think say tell "
        "open close start stop run walk sit stand rise fall move turn meet "
        "play look feel seem appear become remain stay live work happen occur "
        "exist continue begin end pass leave enter return arrive depart "
        "wait rest hold keep let put set find show call ask try want need "
        "like love hate hope wish believe mean matter "
        "grow change vary differ relate concern involve")
    for w in NOISE.split():
        lexicon.pop(w, None)
    json.dump(lexicon, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=0, sort_keys=True)
    print(f"causal verb lexicon: {len(lexicon)} verbs -> {OUT}")
    # quick polarity tally
    from collections import Counter
    tally = Counter(lexicon.values())
    print("  by polarity:", dict(tally))


if __name__ == "__main__":
    main()
