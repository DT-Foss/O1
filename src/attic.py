#!/usr/bin/env python3 -u
"""
ATTIC — the one door to the .causal knowledge index (the KNOWLEDGE node of the pathfinding machine).
====================================================================================================
Bundles the whole vendored fabel pipeline behind a single clean interface so the GSSM side never
touches fabel internals. Three capabilities, one object:

    attic = Attic.from_corpus(text)        # build a fresh .causal index from raw text (LLM-free)
    attic = Attic.from_file(path)          # or load an existing .causal
    attic.knows(term)  -> bool             # does the index cover this gap?
    attic.lookup(term) -> [paths]          # 1-hop facts mentioning the term (with provenance)
    attic.reason(term) -> [chains]         # MULTI-HOP transitive paths via the RICH engine
                                           #   (transitive closure ≤5 hops — the real amplification,
                                           #    ~6x / 18x more connections than the 3-pass stub)

Everything is vendored under vendor/fabel (FORGE excluded). The GSSM streaming state consults this
attic ONLY on a gap (high surprise); the attic answers from measured structure, zero hallucination,
with sources. This is the "attic / second file" David specified — external to the O(1) state.
"""
import os, sys, re

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FABEL = os.path.join(_ROOT, "vendor", "fabel")
for p in (os.path.join(_FABEL, "dotcausal_package", "src"),
          os.path.join(_FABEL, "extract"),
          os.path.join(_FABEL, "language", "hsslm_s")):
    if p not in sys.path:
        sys.path.insert(0, p)

from dotcausal import CausalReader, CausalWriter          # noqa: E402
import rule_extractor as _rx                              # noqa: E402
import inference as _rich                                 # noqa: E402  (the rich transitive engine)

_CONF = {"high": 0.9, "medium": 0.7, "low": 0.5}


def _sentences(text):
    for s in re.split(r"(?<=[.!?])\s+", text):
        s = s.strip()
        if 20 < len(s) < 600:
            yield s


class Attic:
    """The knowledge index, consulted on a gap. Wraps the vendored fabel engine read-side, plus
    the rich transitive-closure inference for multi-hop reasoning."""

    def __init__(self, reader: CausalReader, path: str = None):
        self.reader = reader
        self.path = path
        self.triplets = reader.get_all_triplets(include_inferred=False)
        # surface-term → triplet index (for fast gap lookup)
        self._by_term = {}
        for t in self.triplets:
            surf = f"{t.get('trigger','')} {t.get('mechanism','')} {t.get('outcome','')}".lower()
            for w in set(re.findall(r"[a-z]{4,}", surf)):
                self._by_term.setdefault(w, []).append(t)
        # entity ↔ id maps + adjacency kb for the rich engine
        self._ent2id, self._id2ent = {}, {}
        self._kb = [(self._gid(t["trigger"]), t["mechanism"], self._gid(t["outcome"]))
                    for t in self.triplets]

    # ---- constructors -------------------------------------------------------
    @classmethod
    def from_file(cls, path):
        return cls(CausalReader(path), path)

    @classmethod
    def from_corpus(cls, text, out_path, api_id="corpus", max_sentences=200000, source="corpus"):
        """Build a fresh .causal index from raw text — deterministic, LLM-free (rule_extractor)."""
        w = CausalWriter(api_id=api_id)
        n = 0
        for i, s in enumerate(_sentences(text)):
            if i >= max_sentences:
                break
            for rt in _rx.extract_from_text(s, domain=api_id, source=source):
                d = rt.__dict__
                if d.get("trigger") and d.get("mechanism") and d.get("outcome"):
                    w.add_triplet(str(d["trigger"]), str(d["mechanism"]), str(d["outcome"]),
                                  confidence=_CONF.get(str(d.get("confidence", "medium")).lower(), 0.7),
                                  source=source, domain=api_id, evidence=s[:200])
                    n += 1
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        w.save(out_path)
        return cls(CausalReader(out_path), out_path), n

    # ---- internals ----------------------------------------------------------
    def _gid(self, s):
        if s not in self._ent2id:
            idx = len(self._ent2id)
            self._ent2id[s] = idx
            self._id2ent[idx] = s
        return self._ent2id[s]

    # ---- the gold-edge interface -------------------------------------------
    def knows(self, term):
        """Does the attic cover this gap term?"""
        return term.lower() in self._by_term

    def lookup(self, term, k=3):
        """1-hop facts mentioning `term`, with provenance (zero hallucination)."""
        out = []
        for h in self._by_term.get(term.lower(), [])[:k]:
            out.append({"path": f"{h['trigger']} → {h['mechanism']} → {h['outcome']}",
                        "conf": h.get("confidence"), "source": (h.get("source") or "")[:48]})
        return out

    def reason(self, term, max_hops=5, k=5):
        """MULTI-HOP transitive chains seeded from entities mentioning `term`. This is the rich
        engine: paths the 1-hop lookup (and the 3-pass stub) cannot reach. Returns chains as
        readable strings with confidence."""
        seeds = [self._gid(t["trigger"]) for t in self._by_term.get(term.lower(), [])]
        if not seeds:
            return []
        paths = _rich.complete_inference_pipeline(
            list(set(seeds)), knowledge_base=self._kb, token_id_to_str=self._id2ent,
            max_path_length=max_hops)
        chains = []
        for path, conf, prov in paths:
            if len(path) > 2:                                  # genuine multi-hop only
                chain = " → ".join(self._id2ent.get(t, "?") for t in path)
                chains.append({"chain": chain, "hops": len(path) - 1, "conf": round(float(conf), 3)})
        chains.sort(key=lambda c: (-c["hops"], -c["conf"]))
        return chains[:k]

    # ---- GROWTH: the attic expands itself ----------------------------------
    def _extract(self, text, source, api_id):
        out = []
        for s in _sentences(text):
            for rt in _rx.extract_from_text(s, domain=api_id, source=source):
                d = rt.__dict__
                if d.get("trigger") and d.get("mechanism") and d.get("outcome"):
                    out.append((str(d["trigger"]), str(d["mechanism"]), str(d["outcome"]),
                                _CONF.get(str(d.get("confidence", "medium")).lower(), 0.7), s[:200]))
        return out

    def add_text(self, text, source="retrieved", api_id=None, flush=True):
        """Grow the attic from new text. flush=True writes now (simple); flush=False STAGES the
        triplets in memory for a later batch flush() — the fast path for many adds (one rewrite of
        the graph instead of one per call). The self-expanding loop: meet a gap, retrieve, fold in,
        next time KNOW. Returns number of new triplets staged/added."""
        api_id = api_id or "grown"
        new = self._extract(text, source, api_id)
        if not new:
            return 0
        if not hasattr(self, "_staged"):
            self._staged = []
        self._staged.extend((t, m, o, c, e, source, api_id) for (t, m, o, c, e) in new)
        if flush:
            self.flush()
        return len(new)

    def flush(self):
        """Write existing + all staged triplets to the .causal ONCE, then reload (inference
        re-materializes at save). Fast path: stage many add_text(flush=False), flush() once."""
        staged = getattr(self, "_staged", [])
        if not staged:
            return 0
        w = CausalWriter(api_id="grown")
        for t in self.triplets:
            w.add_triplet(t["trigger"], t["mechanism"], t["outcome"],
                          confidence=t.get("confidence", 0.7), source=t.get("source", ""),
                          domain=t.get("domain", ""), evidence=t.get("evidence", ""))
        for (trig, mech, outc, conf, ev, src, api) in staged:
            w.add_triplet(trig, mech, outc, confidence=conf, source=src, domain=api, evidence=ev)
        n = len(staged)
        self._staged = []
        if self.path:
            w.save(self.path)
            self.__init__(CausalReader(self.path), self.path)
        return n

    def stats(self):
        return {"path": self.path, "triplets": len(self.triplets),
                "entities": len(self._ent2id), "indexed_terms": len(self._by_term)}


if __name__ == "__main__":
    # smoke: build from WikiText, then show 1-hop vs multi-hop on a term
    sys.path.insert(0, os.path.join(_ROOT, "src"))
    from length_extrap_v2 import load_wikitext2
    text, _ = load_wikitext2()
    attic, n = Attic.from_corpus(text, os.path.join(_ROOT, "demo/causal_assets/wikitext.causal"),
                                 api_id="wikitext", source="wikitext-2")
    print(f"[attic] built from WikiText: {attic.stats()} ({n} triplets extracted)")
    for term in ["damage", "increased", "caused"]:
        if attic.knows(term):
            hop1 = attic.lookup(term, k=1)
            multi = attic.reason(term, k=3)
            print(f"\n  '{term}': {len(hop1)} 1-hop, {len(multi)} multi-hop chains")
            for c in multi[:2]:
                print(f"     [{c['hops']}hop {c['conf']}] {c['chain'][:100]}")
