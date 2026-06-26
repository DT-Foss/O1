"""
demo.py — the "looks like a normal AI chat, then BAM" harness.

It opens like any chat: ask it questions, it answers in plain language. The
twist, revealed on demand, is that there is NO transformer behind it — every
answer comes from a deterministic causal graph on a single CPU core, no GPU, no
network, and the engine fits in tens of MB. Then the reveal: point it at a
folder of PDFs and it chews the whole corpus into facts and proposes cross-paper
hypotheses no single paper states — the thing a language model cannot do
honestly, done with zero tokens and zero fabrication.

Commands:
    <question>            ask the loaded graph (what causes X / how does X lead to Y / ...)
    :ingest PATH         add a paper/folder to the live graph (pdf/txt/md)
    :hypothesize         generate cross-paper hypotheses over the ingested corpus
    :controversies       find where the ingested papers DISAGREE (τ non-contraction)
    :reveal              print the meta-levers (what just happened, and the cost)
    :topics  :q
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fabel import Fabel, GraphIndex          # noqa: E402
import hypothesize as H                       # noqa: E402


BANNER = """\
┌──────────────────────────────────────────────────────────────┐
│  assistant ready.  ask me anything about what's loaded.       │
│  (try :reveal once you've seen a few answers)                 │
└──────────────────────────────────────────────────────────────┘"""

REVEAL = """\
═══ what you were actually talking to ═══════════════════════════
  • NOT a transformer. No GPU, no network, no tokens spent.
  • Every answer is a real edge in a causal graph, with provenance.
    An answer that isn't in the graph CANNOT be produced — the
    zero-hallucination is structural, not statistical.
  • Engine + graph fit in tens of MB. Runs fully air-gapped.
  • Answers in microseconds on ONE core (you saw the timings).
  • The corpus → facts → cross-paper hypotheses pipeline ran with
    zero language-model calls. The leads it proposes are assembled
    from facts two papers state separately and neither connects.
═════════════════════════════════════════════════════════════════"""


class Demo:
    def __init__(self, graph_path: str | None = None):
        self.index = GraphIndex(graph_path) if graph_path else GraphIndex()
        self.bot = Fabel(graph=self.index)
        self.corpus: list = []        # ingested paper paths (for hypothesizing)
        self.answered = 0

    def ingest(self, path: str) -> str:
        paths = []
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for f in files:
                    if f.endswith((".pdf", ".txt", ".md")):
                        paths.append(os.path.join(root, f))
        elif os.path.isfile(path):
            paths = [path]
        else:
            return f"no such path: {path}"
        # build facts into the live graph via the standard extract→build path
        t0 = time.time()
        H.rx.set_concept_mode(False)   # graph answering keeps full-fidelity entities
        from extract_to_db import _read_text
        added = 0
        for p in paths:
            text = _read_text(p)
            for t in H.rx.extract_from_text(text, source=os.path.basename(p)):
                a = self.index._sym(t.trigger.lower())
                b = self.index._sym(t.outcome.lower())
                if a is None or b is None or a == b:
                    continue
                self.index.fwd.setdefault(a, {})[b] = H._conf(t.confidence)
                self.index.rev.setdefault(b, {})[a] = H._conf(t.confidence)
                self.index.mech[(a, b)] = " ".join(t.mechanism.lower().split())
                self.index.meta[(a, b)] = (H._conf(t.confidence),
                                           os.path.basename(p), False, "ingest")
                self.index.n_explicit += 1
                added += 1
        self.corpus.extend(paths)
        dt = time.time() - t0
        return (f"ingested {len(paths)} paper(s) → {added} new facts "
                f"in {dt:.1f}s ({self.index.n_explicit} total). no GPU, no LLM.")

    def controversies(self, top: int = 6) -> str:
        if not self.corpus:
            return "nothing ingested yet — try :ingest PATH first."
        import time
        import hypothesize as _H
        from controversy import detect
        t0 = time.time()
        edges = _H._recanon_with_source(_H._edges_from_corpus(self.corpus))
        cons = detect(edges)
        dt = time.time() - t0
        if not cons:
            return (f"no cross-paper controversies in {len(self.corpus)} papers "
                    f"({dt:.1f}s). Either the corpus agrees, or it's too small for "
                    f"two papers to directly contradict — needs density.")
        out = [f"\n  {len(cons)} cross-paper disagreements found in {dt:.1f}s, "
               f"zero LLM:\n"]
        for c in cons[:top]:
            out.append(f"  ⚔ {c.verbalize()}\n")
        return "\n".join(out)

    def hypothesize(self, top: int = 6) -> str:
        if not self.corpus:
            return "nothing ingested yet — try :ingest PATH first."
        t0 = time.time()
        hyps = H.generate(self.corpus)
        dt = time.time() - t0
        if not hyps:
            return "no concept-bridged cross-paper hypotheses in this corpus."
        out = [f"\n  chewed {len(self.corpus)} papers → {len(hyps)} cross-paper "
               f"hypotheses in {dt:.1f}s, zero LLM calls\n"]
        for i, h in enumerate(hyps[:top], 1):
            out.append(f"  ⟂ HYPOTHESIS {i} (surprise {h.surprise:.3f})")
            out.append(f"    {h.verbalize()}\n")
        return "\n".join(out)

    def ask(self, q: str) -> str:
        t0 = time.time()
        ans = self.bot.answer(q)
        us = (time.time() - t0) * 1e6
        self.answered += 1
        return f"{ans}\n  ⏱ {us:,.0f} µs · 1 core · no GPU"


def main() -> None:
    graph = sys.argv[1] if len(sys.argv) > 1 else None
    d = Demo(graph)
    print(BANNER)
    if graph:
        print(f"  (loaded {len(d.index.vocab)} concepts from "
              f"{os.path.basename(graph)})")
    print()
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
        if q == ":reveal":
            print(REVEAL + "\n")
            continue
        if q == ":topics":
            print("fabel > " + ", ".join(d.index.topics(12)) + "\n")
            continue
        if q.startswith(":ingest "):
            print("fabel > " + d.ingest(q[8:].strip()) + "\n")
            continue
        if q == ":hypothesize":
            print("fabel > " + d.hypothesize() + "\n")
            continue
        if q == ":controversies":
            print("fabel > " + d.controversies() + "\n")
            continue
        print("fabel > " + d.ask(q) + "\n")


if __name__ == "__main__":
    main()
