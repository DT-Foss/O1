"""
speak_mined.py — speak_causal with a MEASURED pattern bank instead of
handwritten connective tables.

Same contract as speak_causal.py: facts exact (entities + mechanism verbs from
the .causal graph, never invented), form generated — but the discourse
connectives now come from a pattern bank mined out of a real corpus
(extract_patterns.py). Determinism split: the FACTS must be deterministic
(graph lookup is), the FORM may be random — connectives are sampled with a
true RNG from the measured count x F30 confidence distribution
(tau-controlled temperature), with a recency penalty so sentences vary.

v1 granularity note: only openers whose tokens are pure connectives are used
as sentence prefixes (an opener like "and if i" needs a clause frame, not a
"SUBJ VERB OBJ" continuation — that is granularity level 2, frame composition).
The bank already stores the frames; this script does not use them yet.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hsslm_s import sampler, state_init
from hsslm_s.pattern_bank import PatternBank
from speak_causal import GRAPH, _NEG_VERBS, _det, load_graph, walk_chain

BANK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faraday_bank.json")

# Connective-only filter: openers usable directly before a "SUBJ VERB OBJ"
# clause. Closed syntactic class, not content.
_SAFE_PREFIX = frozenset(
    "and but so now then therefore thus however yet hence consequently "
    "moreover also indeed again still accordingly nevertheless furthermore "
    "besides".split())
# adverbial connectives that read better with a following comma
_COMMA_AFTER = frozenset(
    "however therefore thus hence consequently now indeed moreover "
    "furthermore nevertheless accordingly again then".split())


class MinedOpeners:
    """Opener choice from the measured inventory. True RNG — form may be
    random, only the graph facts must be deterministic."""

    def __init__(self, bank: PatternBank, seed: int | None = None):
        self.pools: dict = {}
        for cls in ("cause", "contrast", "add"):
            inv = [o for o in bank.opener_inventory(polarity=cls)
                   if all(t in _SAFE_PREFIX for t in o["tokens"])]
            self.pools[cls] = inv
        # cause pool is thin in small corpora — additive connectives carry
        # causal-neutral continuation, merge them in as backoff
        self.pools["cause"] = self.pools["cause"] + self.pools["add"]
        self.recent: list = []
        self.rng = np.random.default_rng(seed)

    def pick(self, polarity: str, tau: float = 0.8) -> str:
        pool = self.pools.get(polarity) or self.pools["add"]
        if not pool:
            return ""
        logits = np.array([np.log(o["count"]) + np.log(o["conf"] + 1e-9)
                           for o in pool])
        # strong recency penalty: a connective used in the last sentences is
        # effectively blocked, variety beats raw frequency within a paragraph
        for k, idx in enumerate(self.recent[-4:]):
            if idx < len(logits):
                logits[idx] -= 4.0 * (k + 1)
        # tau -> temperature as in the contraction sampler, but sampled with
        # a real RNG instead of the deterministic hash
        T = max(sampler.tau_to_temperature(tau), 1e-6)
        probs = np.exp((logits - logits.max()) / T)
        probs /= probs.sum()
        choice = int(self.rng.choice(len(pool), p=probs))
        self.recent.append(choice)
        toks = pool[choice]["tokens"]
        text = " ".join(toks)
        if toks[-1] in _COMMA_AFTER:
            text += ","
        return text


def _np(entity: str) -> str:
    """Article only for short noun phrases; long extracted phrases stay bare."""
    return _det(entity) if len(entity.split()) <= 3 else entity


def _clause_tail(outcome: str) -> str:
    """Attach an outcome to a clause-length mechanism, by outcome shape."""
    first = outcome.split()[0]
    if first.endswith("ing"):
        return f"thereby {outcome}"           # "producing tb of raw data"
    if first.endswith("s") and not first.endswith(("ss", "us", "is")):
        return f"which {outcome}"             # "increases duty cycle"
    return f"the result is {outcome}"         # noun-phrase outcome


def verbalize_mined(hops, vocab, mech, openers: MinedOpeners,
                    tau: float = 0.8) -> str:
    # form tau is deliberately loose (0.8): the facts are pinned by the graph,
    # so the connective choice can explore — that is where fluency variety
    # comes from
    """Polarity-propagating verbalization with mined connectives.

    Two mechanism shapes from real graphs: short verbs ("causes") take the
    SUBJ VERB OBJ frame; clause-length mechanisms (scientific extraction,
    e.g. DZA) take SUBJ MECH-CLAUSE — TAIL(outcome)."""
    if not hops:
        return "(no causal path from there.)"
    sents = []
    prev_neg = False
    for i, (a, b) in enumerate(hops):
        subj, obj = vocab[a], vocab[b]
        verb = mech.get((a, b), "leads to")
        cur_neg = bool(set(verb.split()) & _NEG_VERBS)
        clause = len(verb.split()) > 2
        op = "" if i == 0 else openers.pick(
            "contrast" if (cur_neg or (prev_neg and not cur_neg)) else "cause",
            tau)
        prefix = f"{op.capitalize()} " if op else ""
        if clause:
            s = f"{prefix}{_np(subj)} {verb} — {_clause_tail(obj)}."
        elif i > 0 and prev_neg and not cur_neg:
            # polarity flip (pass2 sign rule): state the relation about the
            # subject itself, do not imply the chain improves the outcome
            s = f"{prefix}{_np(subj)} is exactly what {verb} {_np(obj)}."
        else:
            s = f"{prefix}{_np(subj)} {verb} {_np(obj)}."
        if not prefix:
            s = s[0].upper() + s[1:]
        sents.append(s)
        prev_neg = cur_neg
    return " ".join(sents)


def main() -> None:
    # usage: speak_mined.py [graph.causal] [start entity ...]
    graph_path = sys.argv[1] if len(sys.argv) > 1 else GRAPH
    starts = sys.argv[2:]
    if not os.path.exists(BANK):
        sys.exit(f"pattern bank missing — run: python3 extract_patterns.py -o {BANK}")
    bank = PatternBank.load(BANK)
    vocab, stoi, adj, mech = load_graph(graph_path)
    SM = state_init.initialize_symbol_state(len(vocab))
    if not starts:
        # no starts given: take the entities with the most outgoing edges
        ranked = sorted(adj.items(), key=lambda kv: -len(kv[1]))[:4]
        starts = [vocab[i] for i, _ in ranked]

    openers = MinedOpeners(bank)
    n_pool = {cls: len(v) for cls, v in openers.pools.items()}
    print(f"graph: {len(vocab)} symbols | bank: {bank.n_sentences} sentences, "
          f"{len(bank.openers)} openers, usable connective pools {n_pool}\n")
    print("=== SPEECH from the .causal graph, form measured from corpus ===\n")
    for start in starts:
        hops = walk_chain(start, vocab, stoi, adj, SM, tau=0.3)
        chain = (" -> ".join([vocab[hops[0][0]]] + [vocab[b] for _, b in hops])
                 if hops else "(dead end)")
        print(f"[{start}]")
        print(f"  chain : {chain}")
        print(f"  speech: {verbalize_mined(hops, vocab, mech, openers)}\n")


if __name__ == "__main__":
    main()
