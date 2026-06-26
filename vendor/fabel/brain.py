"""
brain.py — the main brain. One base .causal for general speaking; domain
graphs mounted modularly; links/docs/folders ingested in bulk into new graphs.

A brain always has a BASE graph loaded (general knowledge + speaking ability).
Domain knowledge is added on demand and removed when no longer needed; every
mounted graph shares ONE entity space, so a causal chain can cross module
boundaries — load 'biology' and 'chemistry' and the brain can answer a question
neither graph could alone (the Paper-A/Paper-B amplification, at module scale).

Ingestion turns a source into a graph and mounts it: a text file, a folder
(walked in bulk), or a URL (fetched → text). Everything is deterministic — the
rule extractor + Foss gate + embedded inference, no LLM anywhere. FORGE-style
code/repo scraping plugs in at the SOURCE_ADAPTERS seam (stubbed, see below).

REPL:
    :mount PATH [as NAME]    mount a .causal as a module
    :unmount NAME            remove a module
    :mounted                 list mounted modules
    :ingest SOURCE [as NAME] file / folder / URL -> .causal -> mount
    :topics                  most-connected entities across all modules
    :save PATH               write the merged brain to one .causal
    :help  :q

    anything else            a question (what causes X / what does X cause /
                             how does X lead to Y / tell me about X)
"""
from __future__ import annotations

import os
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "dotcausal_package", "src"))
# the loop/ bridge layer (forge_adapter lives there in the monorepo); optional
_LOOP = os.path.join(os.path.dirname(HERE), "loop")
if os.path.isdir(_LOOP):
    sys.path.insert(0, _LOOP)

from fabel import Fabel
from federation import Federation
from character import Character
from dotcausal import CausalWriter

DEFAULT_BASE = os.path.join(HERE, "graphs", "base.causal")
GRAPHS_DIR = os.path.join(HERE, "graphs")


# --------------------------------------------------------------- ingestion

def _strip_html(html: str) -> str:
    import re
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = re.sub(r"&[a-z]+;", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _fetch_url(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "fabel-brain/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    return _strip_html(raw) if "<" in raw[:2000] else raw


def ingest(source: str, name: str) -> str:
    """source (file / dir / URL) -> a .causal file path, deterministically.

    Reuses the pipeline: rule extractor + Foss gate -> SQLite -> build .causal.
    Returns the path of the built graph.
    """
    db = os.path.join(GRAPHS_DIR, f"{name}.db")
    causal = os.path.join(GRAPHS_DIR, f"{name}.causal")
    extract = os.path.join(HERE, "extract", "extract_to_db.py")
    build = os.path.join(HERE, "build", "build_causal_from_db.py")

    if source.startswith(("http://", "https://")):
        text = _fetch_url(source)
        tmp = os.path.join(GRAPHS_DIR, f"{name}_fetched.txt")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
        corpus = tmp
    else:
        corpus = source  # file or directory; extract_to_db walks dirs

    for p in (db, causal):
        if os.path.exists(p):
            os.remove(p)
    subprocess.run([sys.executable, extract, corpus, "--db", db,
                    "--domain", name], check=True, cwd=HERE)
    subprocess.run([sys.executable, build, "--db", db, "-o", causal,
                    "--quiet"], check=True, cwd=HERE)
    return causal


# Source adapters: each turns a source into a canonical .causal path.
#   text  — file / folder / URL via the deterministic rule pipeline
#   forge — FORGE's own code knowledge (GitHub/repos/languages it already
#           scraped into triplets), transcoded from its format. FORGE stays
#           the engine; this just reads what it produced.
try:
    from forge_adapter import adapter as _forge_adapter
    SOURCE_ADAPTERS = {"text": ingest, "forge": _forge_adapter}
except Exception:
    SOURCE_ADAPTERS = {"text": ingest}


# --------------------------------------------------------------------- brain

class Brain:
    def __init__(self, base_path: str | None = None):
        # federated: modules stay distinct, bridged only at shared entities
        self.g = Federation()
        self.base_loaded = False
        if base_path and os.path.exists(base_path):
            n = self.g.add_graph(base_path, module="base")
            self.base_loaded = n > 0
        # fabel answering machinery, pointed at the federation view
        self.bot = Fabel(graph=self.g)
        # character: stable identity + decaying memory (never merged into facts)
        self.char = Character(os.path.join(GRAPHS_DIR, "character.json"))
        forgotten = self.char.age()   # apply the forgetting curve at startup
        self.forgotten = forgotten

    # ------------------------------------------------------------- modules
    def mount(self, path: str, name: str, include_inferred: bool = True) -> str:
        if not os.path.exists(path):
            return f"no such graph: {path}"
        n = self.g.add_graph(path, module=name, include_inferred=include_inferred)
        return f"mounted '{name}': +{n} explicit edges " \
               f"({len(self.g.vocab)} entities total)"

    def unmount(self, name: str) -> str:
        if name == "base":
            return "refusing to unmount the base module"
        n = self.g.remove_module(name)
        return f"unmounted '{name}': -{n} edges" if n else f"no module '{name}'"

    def mounted(self) -> str:
        if not self.g.modules:
            return "(nothing mounted)"
        rows = [f"  {nm:14s} {info['n']:>5} edges   {info['path']}"
                for nm, info in self.g.modules.items()]
        return "\n".join(rows)

    def ingest_and_mount(self, source: str, name: str) -> str:
        try:
            causal = ingest(source, name)
        except subprocess.CalledProcessError as exc:
            return f"ingest failed: {exc}"
        except Exception as exc:
            return f"ingest error: {type(exc).__name__}: {exc}"
        return f"ingested {source}\n" + self.mount(causal, name)

    def ingest_forge(self, source: str, name: str = "forge",
                     rebuild: bool = False) -> str:
        """Mount FORGE's code knowledge. Transcoding 48K triplets is slow, so a
        prior graphs/<name>.causal is reused unless rebuild=True."""
        if "forge" not in SOURCE_ADAPTERS:
            return "forge adapter unavailable (msgpack/zlib or forge_adapter missing)"
        cached = os.path.join(GRAPHS_DIR, f"{name}.causal")
        # large FORGE KBs: inference is expensive and only persisted if small
        # enough; skip live inference at mount (persisted edges still load).
        if os.path.exists(cached) and not rebuild:
            return (f"mounting cached FORGE graph (use ':forge rebuild' to "
                    f"re-transcode)\n"
                    + self.mount(cached, name, include_inferred=False))
        from forge_adapter import DEFAULT_FORGE_KB
        src = source or DEFAULT_FORGE_KB
        if not os.path.exists(src):
            return (f"no FORGE knowledge at {src}. Set FORGE_KB or copy a "
                    f"knowledge/ dir into ./forge_kb/ (see TODO.md). FORGE is "
                    f"an optional, deferred integration.")
        try:
            causal = SOURCE_ADAPTERS["forge"](src, name)
        except Exception as exc:
            return f"forge ingest error: {type(exc).__name__}: {exc}"
        source = src
        return (f"transcoded FORGE knowledge from {source}\n"
                + self.mount(causal, name, include_inferred=False))

    def save(self, path: str) -> str:
        """Flatten the whole merged brain (all modules) into one .causal."""
        w = CausalWriter(api_id="brain")
        for (a, b), meta in self.g.meta.items():
            if meta[2]:          # skip inferred — they re-derive on load
                continue
            w.add_triplet(trigger=self.g.vocab[a], mechanism=self.g.mech[(a, b)],
                          outcome=self.g.vocab[b], confidence=meta[0],
                          source=meta[1], domain=meta[3] if len(meta) > 3 else "")
        stats = w.save(path)
        return f"saved brain -> {path} ({stats['triplets']} explicit triplets)"


def _repl(brain: Brain) -> None:
    mods = ", ".join(brain.g.modules) or "(empty)"
    print(f"brain — {len(brain.g.vocab)} entities, modules: {mods}")
    print("  :mount :unmount :mounted :ingest :topics :save :q\n")
    while True:
        try:
            line = input("brain > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in (":q", ":quit", ":exit"):
            brain.char.save()       # persist memory on the way out
            break
        if line == ":help":
            print(__doc__)
            continue
        if line == ":persona":
            print("  " + brain.char.persona() + "\n")
            continue
        if line == ":memory":
            print("  " + brain.char.summary())
            for m in brain.char.recall(n=8):
                print(f"    · {m}")
            print()
            continue
        if line.startswith(":identity "):
            brain.char.remember_identity(line[10:].strip())
            brain.char.save()
            print("  identity updated\n")
            continue
        if line == ":mounted":
            print(brain.mounted() + "\n")
            continue
        if line == ":topics":
            print("  " + ", ".join(brain.g.topics(12)) + "\n")
            continue
        if line == ":bridges":
            br = brain.g.bridges()
            if not br:
                print("  (no cross-module bridges yet)\n")
            else:
                rows = [f"  {ent:40s} {', '.join(mods)}" for ent, mods in br[:15]]
                print("\n".join(rows) + "\n")
            continue
        if line.startswith(":mount "):
            rest = line[7:].strip()
            if " as " in rest:
                path, name = rest.rsplit(" as ", 1)
            else:
                path = rest
                name = os.path.splitext(os.path.basename(path))[0]
            print(brain.mount(path.strip(), name.strip()) + "\n")
            continue
        if line.startswith(":unmount "):
            print(brain.unmount(line[9:].strip()) + "\n")
            continue
        if line.startswith(":ingest "):
            rest = line[8:].strip()
            if " as " in rest:
                source, name = rest.rsplit(" as ", 1)
            else:
                source = rest
                base = os.path.basename(source.rstrip("/")) or "ingested"
                name = os.path.splitext(base)[0].replace(".", "_") or "ingested"
            print(brain.ingest_and_mount(source.strip(), name.strip()) + "\n")
            continue
        if line.startswith(":forge"):
            rest = line[6:].strip()
            rebuild = rest == "rebuild"
            src = "" if (not rest or rebuild) else rest
            # default comes from forge_adapter (FORGE_KB env or local forge_kb/)
            print(brain.ingest_forge(src, rebuild=rebuild) + "\n")
            continue
        if line.startswith(":save "):
            print(brain.save(line[6:].strip()) + "\n")
            continue
        # otherwise: a question, answered across all mounted modules.
        # observe the utterance for memory (gated — only salient ones stick;
        # this shapes recall/tone, never the factual graphs).
        brain.char.observe(line)
        print(brain.bot.answer(line) + "\n")


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE
    brain = Brain(base)
    if not brain.base_loaded:
        print(f"note: no base graph at {base} — starting empty. "
              f"Mount or ingest something.\n")
    _repl(brain)


if __name__ == "__main__":
    main()
