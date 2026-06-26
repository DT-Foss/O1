"""
character.py — a brain's character: a stable identity + a decaying memory.

The danger with "the brain remembers things" is poisoning: if it writes its own
statements back as facts and later reads them as evidence, confidence compounds
falsely until it "knows" things it only told itself. The defense, built in here:

  1. SEPARATION. Two graphs, never merged into domain knowledge:
       - identity : small, stable, curated (who the brain is, its stance).
       - episodic : auto-written, gated, CAPPED, and DECAYING.
     Both live under provenance class '@memory' / '@identity' and never feed
     factual inference — they shape recall and tone, not truth.

  2. DECAY = forgetting curve. Every episodic memory carries a strength that
     ages each session. Below a floor it is dropped. So memory is "some, not
     too much" by construction — the cap is strength×age, not a fixed count.

  3. WRITE GATE. Not every interaction is remembered; only salient ones
     (a stated preference, a correction, a recurring topic). Trivia is dropped.

Memory is stored as plain JSON next to the brain, not as causal triplets, so it
can never be confused with or merged into the knowledge graphs.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

DECAY = 0.85          # per-session strength multiplier (the forgetting curve)
FLOOR = 0.15          # below this, a memory is forgotten
CAP = 60              # hard ceiling on episodic memories (oldest-weakest drop)
WRITE_MIN_LEN = 12    # don't remember trivially short utterances


class Character:
    def __init__(self, path: str):
        self.path = path
        self.identity: List[str] = []     # stable, curated lines
        self.episodic: List[Dict] = []    # [{"text","strength","topic"}]
        self.name = "fabel"
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, encoding="utf-8") as fh:
            d = json.load(fh)
        self.name = d.get("name", "fabel")
        self.identity = d.get("identity", [])
        self.episodic = d.get("episodic", [])

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({"name": self.name, "identity": self.identity,
                       "episodic": self.episodic}, fh, ensure_ascii=False,
                      indent=2)

    # ----------------------------------------------------------- lifecycle
    def age(self) -> int:
        """Apply the forgetting curve once (call at session start). Returns the
        number of memories forgotten."""
        before = len(self.episodic)
        for m in self.episodic:
            m["strength"] *= DECAY
        self.episodic = [m for m in self.episodic if m["strength"] >= FLOOR]
        # enforce the cap: keep the strongest CAP memories
        self.episodic.sort(key=lambda m: -m["strength"])
        self.episodic = self.episodic[:CAP]
        return before - len(self.episodic)

    # --------------------------------------------------------------- write
    def _salient(self, text: str) -> bool:
        if len(text) < WRITE_MIN_LEN:
            return False
        t = text.lower()
        # remember stated preferences, corrections, identity-ish statements
        cues = ("i like", "i prefer", "i want", "remember", "my ",
                "i think", "i believe", "actually", "no,", "call me",
                "i am", "i'm", "don't", "always", "never")
        return any(c in t for c in cues)

    def observe(self, text: str, topic: str = "") -> bool:
        """Maybe remember an utterance. Returns True if it was stored."""
        text = text.strip()
        if not self._salient(text):
            return False
        # reinforce if we already have a near-duplicate (strengthen, don't dupe)
        for m in self.episodic:
            if m["text"].lower() == text.lower():
                m["strength"] = min(1.0, m["strength"] + 0.2)
                return True
        self.episodic.append({"text": text, "strength": 1.0, "topic": topic})
        return True

    def remember_identity(self, line: str) -> None:
        """Add a stable identity line (curated, does not decay)."""
        line = line.strip()
        if line and line not in self.identity:
            self.identity.append(line)

    # ---------------------------------------------------------------- read
    def recall(self, topic: str = "", n: int = 3) -> List[str]:
        """Strongest relevant memories — for coloring tone/recall, NOT facts."""
        pool = self.episodic
        if topic:
            tl = topic.lower()
            scored = sorted(pool, key=lambda m: (
                -(tl in m["text"].lower() or tl in m.get("topic", "").lower()),
                -m["strength"]))
        else:
            scored = sorted(pool, key=lambda m: -m["strength"])
        return [m["text"] for m in scored[:n]]

    def persona(self) -> str:
        """One-line self-description from the identity graph."""
        if not self.identity:
            return f"I am {self.name}."
        return f"I am {self.name}. " + " ".join(self.identity[:3])

    def summary(self) -> str:
        return (f"{self.name}: {len(self.identity)} identity lines, "
                f"{len(self.episodic)} memories "
                f"(strength {sum(m['strength'] for m in self.episodic):.1f} total)")
