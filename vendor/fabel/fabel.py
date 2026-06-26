"""
fabel — talk to a .causal knowledge graph.

The architecture separates knowledge from language form:

  FACTS    : a .causal graph (dotcausal, deterministic embedded inference).
             Every entity and mechanism in an answer comes from the graph —
             nothing is invented, every fact carries confidence + source.
  FORM     : measured language patterns (hsslm_s pattern bank mined from a
             corpus) + clause-aware templates. May be random; facts may not.
  DIALOGUE : deterministic intent patterns over the question (what causes X /
             what does X cause / how does X lead to Y / tell me about X),
             answered by reverse lookup, forward walk, or BFS path search.

Usage:
    python3 fabel.py [graph.causal]            # interactive REPL
    echo "what causes lung damage?" | python3 fabel.py [graph.causal]

REPL commands: :topics, :load PATH, :help, :q
"""
from __future__ import annotations

import os
import re
import sys
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
# bundled canonical dotcausal package
sys.path.insert(0, os.path.join(HERE, "dotcausal_package", "src"))
# bundled hsslm_s symbolic language modules (mined sentence form). If the
# import fails for any reason, fabel falls back to plain connective phrasing.
sys.path.insert(0, os.path.join(HERE, "language"))

from dotcausal import CausalReader
import typed_inference as ti

try:
    from hsslm_s.inference import jaro_winkler
    from hsslm_s.pattern_bank import PatternBank
    from speak_mined import BANK, MinedOpeners, _clause_tail, _np, verbalize_mined
    _HAS_FORM = True
except Exception:
    _HAS_FORM = False

    def jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
        """Minimal Jaro-Winkler fallback for entity resolution."""
        if s1 == s2:
            return 1.0
        a, b = set(s1.split()), set(s2.split())
        return (2 * len(a & b) / (len(a) + len(b))) if (a or b) else 0.0

    def _np(e: str) -> str:
        return e

    def _clause_tail(o: str) -> str:
        return f"the result is {o}"

DEFAULT_GRAPH = os.path.join(HERE, "graphs", "faraday.causal")


def _toks(s: str):
    return re.findall(r"[a-z]+", str(s).lower())


def _norm(s: str) -> str:
    return " ".join(_toks(s))


# typed semantic relations carry their type in the mechanism slot and must NOT
# be word-normalized ("is-a" -> "is a" would lose the type)
_SEM_MECH = re.compile(r"^(is-a|has-a|part-of|property|defines|does:)")


def _norm_mech(mech: str) -> str:
    """Normalize a mechanism, but preserve typed semantic relations verbatim."""
    m = str(mech).strip()
    if _SEM_MECH.match(m):
        return m
    return _norm(m) or "leads to"


class GraphIndex:
    """Entity index + adjacency over one or more .causal graphs.

    Multiple graphs merge into ONE entity space (same entity string = same
    node), so causal paths can cross module boundaries. Each edge remembers
    which module ('base', a domain name, ...) it came from, for provenance
    and for unmounting.
    """

    def __init__(self, path: str | None = None, module: str = "base"):
        self.vocab: list = []
        self.stoi: dict = {}
        self.fwd: dict = {}      # a -> {b: conf}
        self.rev: dict = {}      # b -> {a: conf}
        self.mech: dict = {}     # (a,b) -> mechanism text
        self.meta: dict = {}     # (a,b) -> (conf, source, is_inferred, module)
        self.attrs: dict = {}    # (a,b) -> typed fields {condition, population,
                                 #          effect_size, co_causes, temporal}
        self.modules: dict = {}  # name -> {"path":..., "edges": set, "n":int}
        self.n_explicit = 0
        if path is not None:
            self.add_graph(path, module)

    def add_graph(self, path: str, module: str = "base") -> int:
        """Merge a .causal graph into the shared space under `module`.
        Returns the number of explicit edges added."""
        trips = CausalReader(path).get_all_triplets()
        edges = self.modules.setdefault(
            module, {"path": path, "edges": set(), "n": 0})["edges"]
        added = 0
        for t in trips:
            a, b = self._sym(t.get("trigger", "")), self._sym(t.get("outcome", ""))
            if a is None or b is None or a == b:
                continue
            c = float(t.get("confidence", 0.5) or 0.5)
            inferred = bool(t.get("is_inferred"))
            # explicit edges win over inferred duplicates of the same pair
            if (a, b) in self.meta and not self.meta[(a, b)][2] and inferred:
                continue
            self.fwd.setdefault(a, {})[b] = max(self.fwd.get(a, {}).get(b, 0), c)
            self.rev.setdefault(b, {})[a] = max(self.rev.get(b, {}).get(a, 0), c)
            self.mech[(a, b)] = _norm_mech(t.get("mechanism", ""))
            self.meta[(a, b)] = (c, t.get("source", "") or "", inferred, module)
            at = t.get("attrs") or {}
            if at:
                self.attrs[(a, b)] = at
            edges.add((a, b))
            if not inferred:
                self.n_explicit += 1
                added += 1
        self.modules[module]["n"] = len(edges)
        return added

    def remove_module(self, module: str) -> int:
        """Unmount a module: drop its edges. Entities stay in vocab (cheap)."""
        info = self.modules.pop(module, None)
        if not info:
            return 0
        for (a, b) in info["edges"]:
            self.fwd.get(a, {}).pop(b, None)
            self.rev.get(b, {}).pop(a, None)
            meta = self.meta.pop((a, b), None)
            self.mech.pop((a, b), None)
            self.attrs.pop((a, b), None)
            if meta and not meta[2]:
                self.n_explicit -= 1
        return len(info["edges"])

    def _sym(self, phrase: str):
        p = _norm(phrase)
        if not p:
            return None
        if p not in self.stoi:
            self.stoi[p] = len(self.vocab)
            self.vocab.append(p)
        return self.stoi[p]

    # -------------------------------------------------------- entity lookup
    def resolve(self, phrase: str):
        """User phrase -> entity id: exact, then containment, then fuzzy."""
        p = _norm(phrase)
        if not p:
            return None
        if p in self.stoi:
            return self.stoi[p]
        # containment (prefer the best-connected entity)
        hits = [i for v, i in self.stoi.items() if p in v or v in p]
        if hits:
            return max(hits, key=lambda i: len(self.fwd.get(i, {}))
                       + len(self.rev.get(i, {})))
        # fuzzy (F49: Jaro-Winkler), threshold from the 3-pass engine
        best: int | None = None
        score = 0.0
        for v, i in self.stoi.items():
            s = jaro_winkler(p, v)
            if s > score:
                best, score = i, s
        return best if score >= 0.80 else None

    def suggest(self, phrase: str, n: int = 5):
        p = _norm(phrase)
        scored = sorted(self.stoi.items(),
                        key=lambda kv: -jaro_winkler(p, kv[0]))[:n]
        return [v for v, _ in scored]

    def topics(self, n: int = 12):
        deg = {i: len(self.fwd.get(i, {})) + len(self.rev.get(i, {}))
               for i in range(len(self.vocab))}
        return [self.vocab[i] for i, _ in
                sorted(deg.items(), key=lambda kv: -kv[1])[:n]]

    def path(self, a: int, b: int, max_depth: int = 6,
             explicit_only: bool = False):
        """Shortest causal path a -> b. With explicit_only, traverse only
        explicit edges so the answer rests on stated facts, not inferred
        shortcuts (BFS shortest-path would otherwise take a weak 1-hop
        inference over a fully-explicit 2-hop chain)."""
        prev: dict = {a: None}
        queue = deque([(a, 0)])
        while queue:
            cur, d = queue.popleft()
            if cur == b:
                hops, node = [], b
                while prev[node] is not None:
                    hops.append((prev[node], node))
                    node = prev[node]
                return list(reversed(hops))
            if d >= max_depth:
                continue
            for nxt in self.fwd.get(cur, {}):
                if nxt in prev:
                    continue
                if explicit_only and self.meta[(cur, nxt)][2]:
                    continue  # skip inferred edge
                prev[nxt] = cur
                queue.append((nxt, d + 1))
        return None


# ---------------------------------------------------------------- answering

class _PlainOpeners:
    """Fallback discourse connectives when the mined pattern bank is absent."""
    _BY = {"cause": "and", "add": "and", "contrast": "but"}

    def pick(self, polarity: str, tau: float = 0.8) -> str:
        return self._BY.get(polarity, "and")


def _verbalize_plain(hops, vocab, mech, openers):
    """Connective-joined verbalization without the mined pattern bank."""
    parts = []
    for i, (a, b) in enumerate(hops):
        verb = mech.get((a, b), "leads to")
        op = "" if i == 0 else openers.pick("add") + " "
        s = f"{op}{vocab[a]} {verb} {vocab[b]}"
        parts.append(s[0].upper() + s[1:] if i == 0 else s)
    return ". ".join(parts) + "."


class Fabel:
    def __init__(self, graph_path: str | None = None, graph=None):
        # accept either a path (single graph) or a prebuilt GraphIndex (the
        # multi-graph brain passes its merged index here)
        self.g = graph if graph is not None else GraphIndex(graph_path)
        if _HAS_FORM and os.path.exists(BANK):
            self.openers = MinedOpeners(PatternBank.load(BANK))
            self._verbalize = verbalize_mined
        else:
            self.openers = _PlainOpeners()
            self._verbalize = _verbalize_plain

    # ---- one fact -> one sentence (same clause-aware shapes as speak_mined)
    def _sentence(self, a: int, b: int, opener: str = "") -> str:
        subj, obj = self.g.vocab[a], self.g.vocab[b]
        verb = self.g.mech[(a, b)]
        prefix = f"{opener.capitalize()} " if opener else ""
        if len(verb.split()) > 2:
            s = f"{prefix}{_np(subj)} {verb} — {_clause_tail(obj)}."
        else:
            s = f"{prefix}{_np(subj)} {verb} {_np(obj)}."
        return s[0].upper() + s[1:] if not prefix else s

    def _provenance(self, edges) -> str:
        lines = []
        for (a, b) in edges:
            meta = self.g.meta[(a, b)]
            c, src, inferred = meta[0], meta[1], meta[2]
            module = meta[3] if len(meta) > 3 else "base"
            kind = "inferred" if inferred else "explicit"
            src = f" | {src}" if src else ""
            mod = f" @{module}" if module and module != "base" else ""
            lines.append(f"    [{c:.2f} {kind}{mod}{src}]  "
                         f"{self.g.vocab[a]} -> {self.g.vocab[b]}")
        return "\n".join(lines)

    # ------------------------------------------------------------- intents
    # weak fuzzy-inferred edges are evidence for gap-finding, not for speech;
    # only facts at/above this confidence are spoken as answers
    SPEAK_MIN_CONF = 0.6
    _SEM_PREFIX = ("is-a", "has-a", "part-of", "property", "defines", "does:")

    def _is_causal(self, a: int, b: int) -> bool:
        """True if the (a,b) edge is a causal mechanism, not a typed semantic
        relation (is-a/has-a/...) — keeps the two answer styles separate."""
        return not self.g.mech.get((a, b), "").startswith(self._SEM_PREFIX)

    def what_causes(self, x: int) -> str:
        causes = sorted(self.g.rev.get(x, {}).items(), key=lambda kv: -kv[1])
        causes = [(a, c) for a, c in causes
                  if (c >= self.SPEAK_MIN_CONF or not self.g.meta[(a, x)][2])
                  and self._is_causal(a, x)][:3]
        if not causes:
            return f"The graph records no firm cause of {_np(self.g.vocab[x])}."
        sents, edges = [], []
        for k, (a, _) in enumerate(causes):
            op = "" if k == 0 else self.openers.pick("add")
            sents.append(self._sentence(a, x, op))
            edges.append((a, x))
        return " ".join(sents) + "\n" + self._provenance(edges)

    def what_follows(self, x: int) -> str:
        if not self.g.fwd.get(x):
            return f"The graph records no consequence of {_np(self.g.vocab[x])}."
        # walk the strongest chain up to 4 hops (facts: deterministic argmax),
        # following only edges firm enough to assert
        hops, cur, seen = [], x, {x}
        for _ in range(4):
            nbrs = {b: c for b, c in self.g.fwd.get(cur, {}).items()
                    if b not in seen and self._is_causal(cur, b)
                    and (c >= self.SPEAK_MIN_CONF
                         or not self.g.meta[(cur, b)][2])}
            if not nbrs:
                break
            nxt = max(nbrs, key=lambda b: nbrs[b])
            hops.append((cur, nxt))
            seen.add(nxt)
            cur = nxt
        if not hops:
            return (f"{_np(self.g.vocab[x]).capitalize()} has only weakly "
                    f"inferred consequences in this graph — no firm chain.")
        # walk the chain with type propagation; if a typed rule prunes a hop
        # (disjoint populations), stop the chain THERE — don't assert past it
        tattrs, broken = self._typed_walk(hops)
        if broken is not None:
            hops = hops[:broken]            # keep only the type-valid prefix
            tattrs, _ = self._typed_walk(hops) if hops else ({}, None)
        if not hops:
            return (f"{_np(self.g.vocab[x]).capitalize()} starts a chain that "
                    f"the typed graph prunes immediately (disjoint populations).")
        prose = self._verbalize(hops, self.g.vocab, self.g.mech, self.openers)
        return prose + self._typed_tail(tattrs) + "\n" + self._provenance(hops)

    # ---- typed chaining over a path: propagate the typed fields hop by hop,
    # using the same rules as typed_inference (condition AND-accumulates,
    # populations intersect — DISJOINT BREAKS THE CHAIN — effect sizes multiply).
    # Returns (accumulated_attrs, broken_at) where broken_at is the hop index a
    # typed rule pruned, or None if the whole chain holds.
    def _typed_walk(self, hops):
        if len(hops) < 2:
            a = self.g.attrs.get(hops[0]) if hops else None
            return (dict(a) if a else {}), None
        # fold the hops left-to-right through typed_inference.chain_two
        acc = {
            "trigger": self.g.vocab[hops[0][0]],
            "outcome": self.g.vocab[hops[0][1]],
            "confidence": self.g.meta[hops[0]][0],
            "attrs": dict(self.g.attrs.get(hops[0], {})),
        }
        for i, hop in enumerate(hops[1:], start=1):
            nxt = {
                "trigger": self.g.vocab[hop[0]],
                "outcome": self.g.vocab[hop[1]],
                "confidence": self.g.meta[hop][0],
                "attrs": dict(self.g.attrs.get(hop, {})),
            }
            chained = ti.chain_two(acc, nxt)
            if chained is None:
                return acc.get("attrs", {}), i   # PRUNED by a typed rule
            acc = chained
        return acc.get("attrs", {}), None

    def _typed_tail(self, attrs: dict) -> str:
        """Render accumulated typed fields as a natural qualifier clause."""
        if not attrs:
            return ""
        bits = []
        if attrs.get("condition"):
            bits.append(f"only when {attrs['condition']}")
        if attrs.get("population"):
            bits.append(f"in {attrs['population']}")
        if attrs.get("effect_size"):
            es = re.sub(r"\s*\(combined\)\s*", "", attrs["effect_size"]).strip()
            bits.append(f"with a combined effect of {es}")
        return ("  (holds " + "; ".join(bits) + ")") if bits else ""

    def how_path(self, x: int, y: int) -> str:
        # prefer a fully-explicit path; fall back to one that uses inferred
        # edges only if no explicit chain exists, and flag it as a lead
        hops = self.g.path(x, y, explicit_only=True)
        weak = False
        if hops is None:
            hops = self.g.path(x, y)
            weak = hops is not None and any(self.g.meta[e][2] for e in hops)
        if hops is None:
            return (f"No causal path from {_np(self.g.vocab[x])} to "
                    f"{_np(self.g.vocab[y])} in this graph (depth <= 5). "
                    f"That absence is itself a finding — a knowledge gap.")
        # type-check the chain: disjoint populations BREAK it (a false-positive
        # path the binary graph would have asserted). Honesty over a wrong yes.
        tattrs, broken = self._typed_walk(hops)
        if broken is not None:
            via = _np(self.g.vocab[hops[broken][0]])
            return (f"A binary path from {_np(self.g.vocab[x])} to "
                    f"{_np(self.g.vocab[y])} exists, but it BREAKS at "
                    f"'{via}': the links hold in disjoint populations, so the "
                    f"chain does not transfer. The typed graph prunes it — "
                    f"this is not a real causal route.")
        prose = self._verbalize(hops, self.g.vocab, self.g.mech, self.openers)
        note = ("\n  (this path rests on weakly inferred links — treat as a "
                "lead, not an established fact)" if weak else "")
        return prose + self._typed_tail(tattrs) + note + "\n" + \
            self._provenance(hops)

    def about(self, x: int) -> str:
        parts = []
        desc = self.describe(x)
        if desc and "knows nothing" not in desc:
            parts.append(desc)
        if self.g.rev.get(x):
            parts.append(self.what_causes(x))
        if self.g.fwd.get(x):
            parts.append(self.what_follows(x))
        return "\n".join(parts) if parts else \
            f"{_np(self.g.vocab[x]).capitalize()} is isolated in this graph."

    # ---- typed semantic relations (is-a / has-a / property / defines / does)
    _SEM_RELS = ("is-a", "has-a", "part-of", "property", "defines")

    def _sem_out(self, x: int, rels):
        """Outgoing semantic edges of x whose relation is in `rels`."""
        out = []
        for b in self.g.fwd.get(x, {}):
            mech = self.g.mech.get((x, b), "")
            base = mech.split(":")[0]
            if base in rels or (rels == ("does",) and mech.startswith("does:")):
                out.append((b, mech))
        return out

    def describe(self, x: int) -> str:
        """Natural description from typed relations: what it is, has, is like."""
        name = _np(self.g.vocab[x])
        bits = []
        isa = self._sem_out(x, ("is-a", "defines"))
        if isa:
            target = self.g.vocab[isa[0][0]]
            art = "an" if target[:1] in "aeiou" else "a"
            bits.append(f"{name.capitalize()} is {art} {target}")
        props = self._sem_out(x, ("property",))
        if props:
            adjs = ", ".join(self.g.vocab[b] for b, _ in props[:3])
            bits.append(f"it is {adjs}")
        has = self._sem_out(x, ("has-a", "part-of"))
        if has:
            parts = ", ".join(_np(self.g.vocab[b]) for b, _ in has[:3])
            bits.append(f"it has {parts}")
        does = self._sem_out(x, ("does",))
        if does:
            mech, b = does[0][1], does[0][0]
            verb = mech.split(":", 1)[1]
            bits.append(f"it {verb} {_np(self.g.vocab[b])}")
        if not bits:
            return f"The graph knows nothing descriptive about {name}."
        return ". ".join(s[0].upper() + s[1:] for s in bits) + "."

    def what_has(self, x: int) -> str:
        has = self._sem_out(x, ("has-a", "part-of"))
        if not has:
            return f"The graph records no parts of {_np(self.g.vocab[x])}."
        parts = ", ".join(_np(self.g.vocab[b]) for b, _ in has)
        return f"{_np(self.g.vocab[x]).capitalize()} has {parts}."

    # --------------------------------------------------------------- router
    INTENTS = [
        # two-entity first: how/why does X lead to / cause / affect Y
        (re.compile(r"(?:how|why)\s+(?:does|do|can|would)?\s*(.+?)\s+"
                    r"(?:lead to|leads to|cause|causes|affect|affects|"
                    r"result in|results in|influence|influences)\s+(.+)",
                    re.I), "path"),
        (re.compile(r"path\s+from\s+(.+?)\s+to\s+(.+)", re.I), "path"),
        (re.compile(r"what\s+(?:causes|leads to|results in|triggers|"
                    r"drives)\s+(.+)", re.I), "causes"),
        (re.compile(r"what\s+(?:does|do)\s+(.+?)\s+have", re.I), "has"),
        (re.compile(r"(?:what\s+(?:does|do)\s+(.+?)\s+(?:cause|do|"
                    r"lead to|trigger)|what\s+happens\s+"
                    r"(?:if|when|with)\s+(.+)|(?:effects?|consequences?)"
                    r"\s+of\s+(.+))", re.I), "follows"),
        # semantic: what is X / describe X (before the generic 'about')
        (re.compile(r"(?:describe|what\s+kind\s+of\s+thing\s+is)\s+(.+)",
                    re.I), "describe"),
        (re.compile(r"(?:tell me about|what is|what's|about|explain)\s+(.+)",
                    re.I), "about"),
    ]

    def answer(self, question: str) -> str:
        q = question.strip().rstrip("?!. ")
        for rx, intent in self.INTENTS:
            m = rx.match(q)
            if not m:
                continue
            groups = [g for g in m.groups() if g]
            ids = []
            for g in groups:
                e = self.g.resolve(g)
                if e is None:
                    sug = ", ".join(self.g.suggest(g))
                    return (f"'{g}' is not in this graph. "
                            f"Closest entities: {sug}")
                ids.append(e)
            if intent == "path" and len(ids) == 2:
                return self.how_path(ids[0], ids[1])
            if intent == "causes":
                return self.what_causes(ids[0])
            if intent == "follows":
                return self.what_follows(ids[0])
            if intent == "has":
                return self.what_has(ids[0])
            if intent == "describe":
                return self.describe(ids[0])
            if intent == "about":
                return self.about(ids[0])
        # bare entity -> about
        e = self.g.resolve(q)
        if e is not None:
            return self.about(e)
        sug = ", ".join(self.g.suggest(q))
        return (f"I can answer: 'what causes X', 'what does X cause', "
                f"'how does X lead to Y', 'tell me about X'. "
                f"Closest entities to your input: {sug}")


def main() -> None:
    graph = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GRAPH
    bot = Fabel(graph)
    g = bot.g
    print(f"fabel — talking to {os.path.basename(graph)}")
    print(f"  {len(g.vocab)} entities, {g.n_explicit} explicit facts, "
          f"{len(g.meta) - g.n_explicit} inferred")
    print(f"  topics: {', '.join(g.topics(6))}")
    print("  ask away (:topics, :load PATH, :q)\n")
    while True:
        try:
            q = input("you   > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in (":q", ":quit", ":exit"):
            break
        if q == ":topics":
            print("fabel > " + ", ".join(g.topics(12)) + "\n")
            continue
        if q.startswith(":load "):
            path = q[6:].strip()
            try:
                bot = Fabel(path)
                g = bot.g
                print(f"fabel > loaded {path}: {len(g.vocab)} entities\n")
            except Exception as exc:
                print(f"fabel > cannot load: {exc}\n")
            continue
        if q == ":help":
            print(__doc__)
            continue
        print("fabel > " + bot.answer(q) + "\n")


if __name__ == "__main__":
    main()
