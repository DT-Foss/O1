"""
federation.py — federated graphs, not one merged blob.

Each .causal stays its own indexed unit (GraphIndex). The Federation holds them
side by side and exposes the SAME interface fabel's answerer expects
(vocab/stoi/fwd/rev/mech/meta + resolve/suggest/topics/path), but builds those
views over a shared ENTITY-BRIDGE index instead of physically merging.

Why federated beats merge:
  - isolation by default: "cell" in a biology graph and "cell" in a battery
    graph stay distinct nodes — no false chain from a string collision.
  - lazy + cheap: graphs load independently, already inferenced on disk.
  - bridging stays possible: a path crosses from graph A to graph B ONLY at an
    entity both graphs literally name (the bridge), and provenance shows the
    hop changed modules. Cross-domain discovery without cross-domain pollution.

The Federation assigns every entity a GLOBAL id but keeps a back-map to
(module, local_entity). Edges live per-module; the merged fwd/rev views are
unions keyed by global id, with each edge tagged by its module in meta.
"""
from __future__ import annotations

import os
import sys
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fabel import GraphIndex, _norm, _norm_mech, jaro_winkler


class Federation:
    """A set of GraphIndex modules sharing a global entity space by NAME.

    Same string (normalized) across modules → same global id → that entity is
    a bridge. Different strings stay different ids → isolation.
    """

    def __init__(self):
        self.vocab: list = []          # global id -> entity string
        self.stoi: dict = {}           # entity string -> global id
        self.fwd: dict = {}            # gid -> {gid: conf}   (union view)
        self.rev: dict = {}            # gid -> {gid: conf}
        self.mech: dict = {}           # (gid,gid) -> mechanism
        self.meta: dict = {}           # (gid,gid) -> (conf, source, inferred, module)
        self.modules: dict = {}        # name -> {"path","edges":set,"n":int}
        self.n_explicit = 0

    def _gid(self, phrase: str):
        p = _norm(phrase)
        if not p:
            return None
        if p not in self.stoi:
            self.stoi[p] = len(self.vocab)
            self.vocab.append(p)
        return self.stoi[p]

    def add_graph(self, path: str, module: str = "base",
                  include_inferred: bool = True) -> int:
        """Load a .causal as its own module; map its entities into the global
        space by name (shared names become bridges).

        include_inferred=False skips the 3-pass inference at load — use for
        huge graphs whose inference wasn't materialized, to keep mount fast.
        A graph that already carries persisted inferred edges returns them
        regardless, at no cost."""
        from dotcausal import CausalReader
        trips = CausalReader(path).get_all_triplets(include_inferred=include_inferred)
        edges = self.modules.setdefault(
            module, {"path": path, "edges": set(), "n": 0})["edges"]
        added = 0
        for t in trips:
            a, b = self._gid(t.get("trigger", "")), self._gid(t.get("outcome", ""))
            if a is None or b is None or a == b:
                continue
            c = float(t.get("confidence", 0.5) or 0.5)
            inferred = bool(t.get("is_inferred"))
            key = (a, b)
            # an explicit edge from any module beats an inferred duplicate
            if key in self.meta and not self.meta[key][2] and inferred:
                continue
            self.fwd.setdefault(a, {})[b] = max(self.fwd.get(a, {}).get(b, 0), c)
            self.rev.setdefault(b, {})[a] = max(self.rev.get(b, {}).get(a, 0), c)
            self.mech[key] = _norm_mech(t.get("mechanism", ""))
            self.meta[key] = (c, t.get("source", "") or "", inferred, module)
            edges.add(key)
            if not inferred:
                self.n_explicit += 1
                added += 1
        self.modules[module]["n"] = len(edges)
        return added

    def remove_module(self, module: str) -> int:
        info = self.modules.pop(module, None)
        if not info:
            return 0
        for key in info["edges"]:
            a, b = key
            self.fwd.get(a, {}).pop(b, None)
            self.rev.get(b, {}).pop(a, None)
            m = self.meta.pop(key, None)
            self.mech.pop(key, None)
            if m and not m[2]:
                self.n_explicit -= 1
        return len(info["edges"])

    # ---- entity bridges: entities that appear in >1 module -----------------
    def bridges(self) -> list:
        """Entities shared across modules — the cross-domain hinge points."""
        by_entity: dict = {}
        for (a, b), meta in self.meta.items():
            mod = meta[3]
            by_entity.setdefault(a, set()).add(mod)
            by_entity.setdefault(b, set()).add(mod)
        return sorted(
            ((self.vocab[gid], sorted(mods)) for gid, mods in by_entity.items()
             if len(mods) > 1),
            key=lambda x: -len(x[1]))

    # ---- the GraphIndex interface fabel's answerer uses --------------------
    def resolve(self, phrase: str):
        p = _norm(phrase)
        if not p:
            return None
        if p in self.stoi:
            return self.stoi[p]
        hits = [i for v, i in self.stoi.items() if p in v or v in p]
        if hits:
            return max(hits, key=lambda i: len(self.fwd.get(i, {}))
                       + len(self.rev.get(i, {})))
        best, score = None, 0.0
        for v, i in self.stoi.items():
            s = jaro_winkler(p, v)
            if s > score:
                best, score = i, s
        return best if score >= 0.80 else None

    def suggest(self, phrase: str, n: int = 5):
        p = _norm(phrase)
        return [v for v, _ in sorted(self.stoi.items(),
                key=lambda kv: -jaro_winkler(p, kv[0]))[:n]]

    def topics(self, n: int = 12):
        deg = {gid: len(self.fwd.get(gid, {})) + len(self.rev.get(gid, {}))
               for gid in range(len(self.vocab))}
        return [self.vocab[i] for i, _ in
                sorted(deg.items(), key=lambda kv: -kv[1])[:n]
                if deg.get(i, 0) > 0]

    def path(self, a: int, b: int, max_depth: int = 6,
             explicit_only: bool = False):
        """BFS over the union view. Crossing modules happens automatically at
        shared (bridge) entities, because a bridge is a single global id with
        edges contributed by multiple modules."""
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
                    continue
                prev[nxt] = cur
                queue.append((nxt, d + 1))
        return None
