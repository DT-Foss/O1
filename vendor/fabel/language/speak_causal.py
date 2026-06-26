"""
speak_causal.py — turn a weight-free .causal walk into natural SPEECH.

run_causal.py proved the symbolic module walks a graph: it emits a correct
causal chain (smoking -> tar buildup -> lung damage -> ...). But a chain is not
speech. This turns the chain into connected sentences, still weight-free:

  - the ENTITIES and MECHANISM verbs are exact, from the graph (never invented).
  - the SENTENCE FORM is generated: articles, subject re-use as pronouns,
    discourse connectives between clauses (therefore / and this / which in turn),
    chosen by a measured, deterministic schedule — no neural net, no training.

So: facts exact (graph), language free (generated form). A long chain is broken
into short clauses and joined with cause/add/contrast connectives so it reads as
prose, not as "A causes B causes C causes D".
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# bundled canonical dotcausal package (two levels up: causal_pipeline/)
sys.path.insert(0, os.path.join(_HERE, "..", "dotcausal_package", "src"))

from hsslm_s import state_init, sampler, bphm
from dotcausal import CausalReader

# default demo graph bundled under causal_pipeline/graphs/
GRAPH = os.path.abspath(os.path.join(_HERE, "..", "graphs", "smoking_demo.causal"))

# discourse connectives by relation polarity — FORM, not fact (same idea as the
# sprache/ lang.py connective table, kept local + weight-free here).
_CAUSE_OPENERS = ["As a result,", "Therefore,", "In turn, this", "This",
                  "Consequently,", "And so", "Which"]
_POS_VERBS = {"improves", "enables", "helps", "increases", "promotes", "boosts"}
_NEG_VERBS = {"prevents", "reduces", "damages", "blocks", "harms", "impairs"}
_CONTRAST = ["However,", "On the other hand,", "Yet"]


def _toks(s):
    return re.findall(r"[a-z]+", str(s).lower())


def load_graph(path):
    trips = CausalReader(path).get_all_triplets()
    vocab, stoi = [], {}

    def sym(phrase):
        p = " ".join(_toks(phrase))
        if p and p not in stoi:
            stoi[p] = len(vocab); vocab.append(p)
        return stoi.get(p)

    adj, mech = {}, {}
    for t in trips:
        a, b = sym(t.get("trigger", "")), sym(t.get("outcome", ""))
        if a is None or b is None:
            continue
        c = float(t.get("confidence", 0.5) or 0.5)
        adj.setdefault(a, {})[b] = max(adj.get(a, {}).get(b, 0), c)
        mech[(a, b)] = " ".join(_toks(t.get("mechanism", ""))) or "leads to"
    return vocab, stoi, adj, mech


def walk_chain(start, vocab, stoi, adj, SM, n=8, tau=0.3):
    """The weight-free walk (same as run_causal): list of (a, b, conf) hops."""
    cur = stoi.get(" ".join(_toks(start)))
    if cur is None:
        return []
    hops, hist = [], []
    for _ in range(n):
        nbrs = adj.get(cur, {})
        if not nbrs:
            break
        logits = np.full(len(vocab), -1e9)
        for b, c in nbrs.items():
            logits[b] = np.log(c + 1e-9)
        nxt = sampler.contraction_sample(logits, tau=tau, top_k=10)
        hist.append(state_init.state_for_symbol(nxt, SM))
        if len(hist) >= 5 and bphm.detect_repetition(hist[-6:]):
            break
        hops.append((cur, int(nxt)))
        cur = int(nxt)
    return hops


def _det(noun):
    """A measured-ish article choice: mass nouns bare, count nouns get 'the'.
    Deterministic + tiny — form, not fact."""
    head = noun.split()[-1]
    mass = {"health", "exercise", "smoking", "sleep", "damage", "stress",
            "breathlessness", "buildup", "caffeine"}
    return noun if head in mass else f"the {noun}"


def verbalize(hops, vocab, mech, seed=0):
    """Chain of hops -> connected prose. First hop is a full clause; later hops
    re-enter with a discourse connective and a pronoun for the carried subject,
    so it reads as speech, not a chain. Connective polarity follows the verb."""
    if not hops:
        return "(no causal path from there.)"
    rng = np.random.RandomState(seed)
    sents = []
    prev_neg = False
    for i, (a, b) in enumerate(hops):
        subj, obj = vocab[a], vocab[b]
        verb = mech.get((a, b), "leads to")
        cur_neg = bool(set(verb.split()) & _NEG_VERBS)
        if i == 0:
            s = f"{_det(subj).capitalize()} {verb} {_det(obj)}."
        elif prev_neg and not cur_neg:
            # polarity flip: the subject was just REDUCED, so a bare
            # "This improves X" would invert the chain's meaning. State the
            # relation about the subject itself instead (pass2 sign rule:
            # (+)+(-) -> (-)).
            s = f"Yet {_det(subj)} is exactly what {verb} {_det(obj)}."
        else:
            # pick a connective by polarity; re-enter with 'this' as the subject
            if cur_neg:
                opener = rng.choice(_CONTRAST)
            else:
                opener = rng.choice(_CAUSE_OPENERS)
            if opener.endswith(("This", "Which")):
                s = f"{opener} {verb} {_det(obj)}."
            else:
                s = f"{opener} {_det(subj)} {verb} {_det(obj)}."
        sents.append(s)
        prev_neg = cur_neg
    # merge very short adjacent clauses with 'and' for flow
    return " ".join(sents)


def main():
    vocab, stoi, adj, mech = load_graph(GRAPH)
    SM = state_init.initialize_symbol_state(len(vocab))
    print(f"graph: {len(vocab)} symbols, {sum(len(v) for v in adj.values())} edges\n")
    print("=== weight-free SPEECH from the .causal graph ===\n")
    for start in ("smoking", "exercise", "caffeine", "lung damage"):
        hops = walk_chain(start, vocab, stoi, adj, SM, tau=0.3)
        print(f"[{start}]")
        chain = (" -> ".join([vocab[hops[0][0]]] + [vocab[b] for _, b in hops])
                 if hops else "(dead end)")
        print(f"  chain : {chain}")
        print(f"  speech: {verbalize(hops, vocab, mech)}\n")


if __name__ == "__main__":
    main()
