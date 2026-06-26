"""
session_memory.py — distill thousands of Claude sessions into a deterministic,
model-agnostic memory graph (a few MB) that any model opens and reads instantly.

This is the SECOND .causal mode: context-oriented, not science-triplet-oriented.
A research paper yields cause→mechanism→outcome; a conversation yields a different
shape — what the user wants, what a project uses, what was decided and why, where
things live. So the relation vocabulary leans on the SEMANTIC extractor (is-a /
has-a / property / does:VERB) plus a few conversational patterns (preference,
decision, location), with causal triplets kept where they genuinely occur.

Every edge is tagged with its session id + project, so cross-session memory works
the way cross-paper hypotheses did: ask "what does the user prefer", "what did we
decide about X", "which sessions touched project Y" — answered from the graph,
instantly, deterministically, no re-reading 3.6 GB of transcripts.

Pipeline: jsonl → clean NL text (drop tool noise) → semantic+causal+conversational
extraction → edges with session provenance → .causal.
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, Iterator, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "extract"))


def _text_of(content) -> str:
    """Pull natural-language text out of a message content (str or block list),
    dropping tool_use / tool_result blocks — the conversational signal, not the
    machinery."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                out.append(b.get("text", ""))
        return "\n".join(out)
    return ""


# lines that are pure mechanics, not memory-bearing prose
_NOISE = re.compile(
    r"^\s*(\$|>|#|```|\||\d+[:.]|import |def |class |return |sudo |cd |ls |git |"
    r"npm |pip |python|/Users/|http|<|\{|\})", re.I)


# harness-injected blocks that are NOT user/assistant prose — stripping these is
# essential: the naive "never/avoid" patterns otherwise grab system-prompt
# boilerplate ("do not recap the summary") as if it were a user preference.
_INJECT = re.compile(
    r"<(system-reminder|command-name|command-message|command-args|"
    r"local-command-caveat|command-stdout|stdout|task-notification|"
    r"new-diagnostics)[\s\S]*?</\1>|"
    r"<[^>]+>|"                                   # any stray tag
    r"^\s*(Caveat:|This session is being continued|<system-reminder)", re.I | re.M)


def _codex_text(content) -> str:
    """Codex content blocks: [{type: input_text|output_text, text: ...}]."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content
                         if isinstance(b, dict) and b.get("type", "").endswith("_text"))
    return ""


def detect_tool(jsonl_path: str) -> str:
    """claude (top-level type=user/assistant) vs codex (type=response_item with a
    nested message payload). Sniffs the first content line."""
    try:
        with open(jsonl_path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("type") == "response_item":
                    return "codex"
                if d.get("type") in ("user", "assistant"):
                    return "claude"
    except Exception:
        pass
    return "claude"


def session_text(jsonl_path: str, max_chars: int = 60000) -> str:
    """Clean NL transcript for EITHER tool: genuine user messages
    (intents/preferences) and assistant prose (decisions/findings). Harness blocks,
    command wrappers, and tool noise removed. Auto-detects Claude vs Codex format."""
    parts: List[str] = []
    tool = detect_tool(jsonl_path)
    try:
        with open(jsonl_path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if tool == "codex":
                    if d.get("type") != "response_item":
                        continue
                    p = d.get("payload") or {}
                    if not isinstance(p, dict) or p.get("type") != "message":
                        continue
                    role = p.get("role")
                    if role not in ("user", "assistant"):
                        continue        # skip developer/system boilerplate
                    txt = _codex_text(p.get("content"))
                else:
                    role = d.get("type")
                    if role not in ("user", "assistant"):
                        continue
                    txt = _text_of(d.get("message", {}).get("content"))
                txt = _INJECT.sub("", txt)        # drop injected boilerplate
                for ln in txt.splitlines():
                    ln = ln.strip()
                    if len(ln) >= 15 and not _NOISE.match(ln):
                        parts.append(f"[{role}] {ln}")
    except Exception:
        return ""
    # the [role] tag lets conversational patterns fire only on USER lines
    return "\n".join(parts)[:max_chars]


# conversational relation patterns — fire ONLY on genuine "[user] " lines, so
# they capture the user's intents/preferences, not harness or assistant text.
_CONV = [
    (re.compile(r"^\[user\].*\b(?:I|we)\s+(?:always|usually|prefer to|like to|"
                r"want to)\s+(.+?)[.;,]", re.I), "prefers"),
    (re.compile(r"^\[user\].*\b(?:decided|chose|went with)\s+(.+?)\s+"
                r"(?:because|since|so that)\s+(.+?)[.;,]", re.I), "decided-because"),
]


def _first_user_intent(text: str) -> str:
    """The session's INTENT = the first substantial [user] line. In a transcript
    this is almost always the task the user opened the session to do — the single
    highest-value memory item per session."""
    skip = ("ja", "ok", "ne ", "nein", "set model", "/", "═", "[david]",
            "this session", "caveat", "<", "you are agent", "du bist agent")
    for ln in text.splitlines():
        if ln.startswith("[user] "):
            body = ln[7:].strip()
            low = body.lower()
            if len(body) >= 20 and not low.startswith(skip):
                return body[:120]
    return ""


# entities worth remembering a session by. Conversational, lightweight:
# CamelCase/acronyms, snake_case, filenames, AND capitalized proper-noun terms in
# prose (so "Nanjing", "Cochrane" survive, not just code identifiers).
_ENTITY = re.compile(
    r"\b([A-Z][A-Za-z0-9]{2,}(?:[A-Z][A-Za-z0-9]+)*|"     # CamelCase / acronyms
    r"[a-z]+(?:_[a-z]+)+|"                                 # snake_case
    r"\w+\.(?:py|md|causal|json|tex|db|pdf)|"              # filenames
    r"[A-Z][a-z]{3,})\b")                                  # capitalized prose noun


def _session_timestamp(jsonl_path: str) -> str:
    """First event's timestamp (YYYY-MM-DD) — lets memory answer 'recently' and
    order sessions in time."""
    try:
        with open(jsonl_path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                try:
                    ts = json.loads(line).get("timestamp")
                except Exception:
                    continue
                if ts:
                    return ts[:10]
    except Exception:
        pass
    return ""


def _iso_week(ts: str) -> str:
    """'2026-06-12' → '2026-W24'. The shared anchor that ties a session to the week
    it belongs to, so 'now' is a node every recent session points at."""
    try:
        import datetime as _dt
        y, m, d = (int(x) for x in ts[:10].split("-"))
        iso = _dt.date(y, m, d).isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"
    except Exception:
        return ""


def _codex_project(jsonl_path: str) -> str:
    """Project for a Codex session = basename of its working dir from session_meta,
    so Codex sessions group by repo like Claude sessions group by project."""
    try:
        with open(jsonl_path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("type") == "session_meta":
                    p = d.get("payload") or {}
                    cwd = (p.get("cwd") or p.get("cwd_path") or "") if isinstance(p, dict) else ""
                    if cwd:
                        return os.path.basename(cwd.rstrip("/"))[:28] or "codex"
                    break
    except Exception:
        pass
    return "codex"


def extract_session(jsonl_path: str, tool: str = "") -> List[Dict]:
    """Context-oriented memory edges for one session: its intent, when it ran, the
    entities it touched, user preferences — tagged with session+project+TOOL.
    `tool` (claude|codex) is auto-detected if empty; it prefixes the session id and
    gets its own edge, so the two tools stay distinguishable in one joint graph."""
    text = session_text(jsonl_path)
    if not text.strip():
        return []
    tool = tool or detect_tool(jsonl_path)
    base = os.path.basename(jsonl_path)
    # codex filenames embed a uuid after the timestamp — use it so each session is
    # unique (the date prefix alone collides across a whole month/project)
    m = re.search(r"([0-9a-f]{8})-[0-9a-f]{4}-[0-9a-f]{4}", base)
    sid = m.group(1) if m else base.replace(".jsonl", "")[:8]
    project = _codex_project(jsonl_path) if tool == "codex" else _project_of(jsonl_path)
    session = f"{tool}:{project}/{sid}"
    edges: List[Dict] = []

    # 0. the tool tag — makes every session filterable by origin
    edges.append({"a": session, "mech": "tool", "b": tool, "conf": 1.0, "src": session})

    # 1. the session's intent (what the user opened it to do)
    intent = _first_user_intent(text)
    if intent:
        edges.append({"a": session, "mech": "intent", "b": intent,
                      "conf": 0.9, "src": session})
    # 2. when it ran (recency) + the ISO-week anchor so all sessions of the same
    # week share a node and "recognize each other as now" (working-memory cohesion)
    ts = _session_timestamp(jsonl_path)
    if ts:
        edges.append({"a": session, "mech": "dated", "b": ts,
                      "conf": 1.0, "src": session})
        week = _iso_week(ts)
        if week:
            edges.append({"a": session, "mech": "in-week", "b": week,
                          "conf": 1.0, "src": session})
    # anchor the session to its project
    edges.append({"a": project, "mech": "has-session", "b": session,
                  "conf": 1.0, "src": session})

    # 3. salient entities (DF-filtered corpus-wide in build_memory; here just
    # within-session frequency). "which sessions used X" works off these.
    from collections import Counter
    ent_freq = Counter()
    for m in _ENTITY.finditer(text):
        e = m.group(1)
        if len(e) >= 3 and e.lower() not in _ENT_STOP:
            ent_freq[e] += 1
    for e, c in ent_freq.most_common(15):
        if c >= 2:                          # mentioned more than once = salient
            edges.append({"a": session, "mech": "mentions", "b": e,
                          "conf": min(0.9, 0.5 + 0.1 * c), "src": session})

    # 4. genuine user preferences (from real [user] lines only)
    for rx_pat, rel in _CONV:
        for ln in text.splitlines():
            m = rx_pat.match(ln)
            if not m:
                continue
            g = [x.strip() for x in m.groups() if x]
            if g and 3 <= len(g[0]) <= 60:
                edges.append({"a": "user", "mech": rel, "b": g[0],
                              "conf": 0.8, "src": session})
    return edges


_ENT_STOP = {"the", "and", "for", "this", "that", "with", "you", "ich", "und",
             "der", "die", "das", "ein", "eine", "claude", "user", "assistant",
             "https", "http", "com", "www", "org", "now", "pass", "team", "users",
             "done", "yes", " pass", "the ", "what", "when", "then", "here", "your",
             "okay", "sure", "also", "just", "let", "lets", "read", "build", "agent",
             "task", "file", "code", "test", "data", "step", "next", "first", "set"}


def build_memory(root: str, out_path: str, df_ceiling: float = 0.12) -> dict:
    """Two-pass build of the session-memory graph with corpus-level DF filtering.

    Pass 1: extract every session's edges and count, for each entity, in how many
    SESSIONS it appears (document frequency). Pass 2: drop entities whose DF
    exceeds `df_ceiling` (ubiquitous = noise, the chat analog of the "result"
    bridge-word problem — an entity in >12% of sessions tells you nothing), then
    write the survivors to one .causal graph.
    """
    import sys as _sys
    from collections import Counter
    _sys.path.insert(0, os.path.join(HERE, "dotcausal_package", "src"))
    from dotcausal import CausalWriter

    sessions = list(iter_sessions(root))
    per_session: List[List[Dict]] = []
    ent_df: Counter = Counter()
    raw_bytes = 0
    for p in sessions:
        try:
            raw_bytes += os.path.getsize(p)
        except OSError:
            pass
        es = extract_session(p)
        per_session.append(es)
        for e in {x["b"] for x in es if x["mech"] == "mentions"}:
            ent_df[e] += 1

    n = max(1, len(sessions))
    ubiquitous = {e for e, c in ent_df.items() if c / n > df_ceiling}

    seen: Dict[tuple, Dict] = {}
    for es in per_session:
        for e in es:
            if e["mech"] == "mentions" and e["b"] in ubiquitous:
                continue                       # DF-filtered noise
            seen[(e["a"], e["mech"], e["b"])] = e

    w = CausalWriter()
    for e in seen.values():
        try:
            w.add_triplet(str(e["a"])[:80], str(e["mech"])[:40],
                          str(e["b"])[:80], confidence=float(e.get("conf", 0.7)))
        except Exception:
            pass
    w.save(out_path)
    return {"sessions": len(sessions), "raw_mb": round(raw_bytes / 1e6, 1),
            "edges": len(seen), "df_filtered_entities": len(ubiquitous),
            "graph_kb": round(os.path.getsize(out_path) / 1024, 1)}


def build_unified(roots: Dict[str, str], out_path: str,
                  df_ceiling: float = 0.12) -> dict:
    """Build ONE session-memory graph from MULTIPLE tools (e.g.
    {'claude': '~/.claude/projects', 'codex': '~/.codex/sessions'}), JOINTLY
    inferenced so a concept shared across tools bridges, while each session stays
    tagged by tool and therefore filterable. Same two-pass DF filtering as
    build_memory, but the document-frequency is computed over the COMBINED corpus
    so cross-tool ubiquity (real noise) is caught, not per-tool.

    Joint (not separate-then-append) is deliberate: appending two independently-
    inferenced graphs leaves two islands — "FORGE" in a Codex session would never
    meet "FORGE" in a Claude session. One graph, by-name bridging, tool tags for
    separability: distinguishable AND connectable."""
    import sys as _sys
    from collections import Counter
    _sys.path.insert(0, os.path.join(HERE, "dotcausal_package", "src"))
    from dotcausal import CausalWriter

    per_session: List[List[Dict]] = []
    ent_df: Counter = Counter()
    raw_bytes = 0
    counts = {t: 0 for t in roots}
    for tool, root in roots.items():
        root = os.path.expanduser(root)
        for p in iter_sessions(root):
            try:
                raw_bytes += os.path.getsize(p)
            except OSError:
                pass
            es = extract_session(p, tool=tool)
            if es:
                counts[tool] += 1
                per_session.append(es)
                for e in {x["b"] for x in es if x["mech"] == "mentions"}:
                    ent_df[e] += 1

    n = max(1, sum(counts.values()))
    ubiquitous = {e for e, c in ent_df.items() if c / n > df_ceiling}

    seen: Dict[tuple, Dict] = {}
    for es in per_session:
        for e in es:
            if e["mech"] == "mentions" and e["b"] in ubiquitous:
                continue
            seen[(e["a"], e["mech"], e["b"])] = e

    w = CausalWriter()
    for e in seen.values():
        try:
            w.add_triplet(str(e["a"])[:80], str(e["mech"])[:40],
                          str(e["b"])[:80], confidence=float(e.get("conf", 0.7)))
        except Exception:
            pass
    w.save(out_path)                       # save runs the joint inference closure
    return {"sessions_by_tool": counts, "total_sessions": n,
            "raw_mb": round(raw_bytes / 1e6, 1), "edges": len(seen),
            "df_filtered_entities": len(ubiquitous),
            "graph_kb": round(os.path.getsize(out_path) / 1024, 1)}


def append_sessions(graph_path: str, roots: Dict[str, str],
                    state_path: str = "") -> dict:
    """Load the persistent graph, append only sessions newer than last time, save.
    One call, terminates — NOT a daemon. The graph stays in its one inferenced
    state; we only add explicit edges (inference is lazy at read). Run it by hand or
    from a cron line; nothing stays running. The ISO-week anchor rides along so the
    current session and its recent predecessors recognize each other as 'now'.

    Returns metrics: sessions/edges added, ms, and tokens this saved vs re-reading
    the new raw history."""
    import sys as _sys, time as _time
    _sys.path.insert(0, os.path.join(HERE, "dotcausal_package", "src"))
    from dotcausal import CausalReader, CausalWriter

    state_path = state_path or os.path.join(
        os.path.dirname(graph_path) or ".", "_mem_state.json")
    last = float(json.load(open(state_path)).get("last_mtime", 0)) \
        if os.path.exists(state_path) else 0.0

    existing = {}
    if os.path.exists(graph_path):
        for t in CausalReader(graph_path, verify_integrity=False).get_all_triplets(
                include_inferred=False):
            existing[(t["trigger"], t["mechanism"], t["outcome"])] = t
    before = len(existing)
    t0 = _time.time()
    newest, n_sess, raw_new = last, 0, 0

    for tool, root in roots.items():
        for p in iter_sessions(os.path.expanduser(root)):
            try:
                mt = os.path.getmtime(p)
            except OSError:
                continue
            if mt <= last:
                continue
            newest = max(newest, mt)
            try:
                raw_new += os.path.getsize(p)
            except OSError:
                pass
            for e in extract_session(p, tool=tool):
                k = (str(e["a"])[:80], str(e["mech"])[:40], str(e["b"])[:80])
                existing[k] = {"trigger": k[0], "mechanism": k[1], "outcome": k[2],
                               "confidence": float(e.get("conf", 0.7))}
            n_sess += 1

    w = CausalWriter()
    for t in existing.values():
        try:
            w.add_triplet(t["trigger"], t["mechanism"], t["outcome"],
                          confidence=t.get("confidence", 0.7))
        except Exception:
            pass
    w.save(graph_path)
    json.dump({"last_mtime": newest}, open(state_path, "w"))
    added = len(existing) - before
    return {"sessions_added": n_sess, "edges_added": added,
            "graph_edges": len(existing), "ms": round((_time.time() - t0) * 1000),
            "tokens_saved_vs_reread": raw_new // 4 - added * 12}


def recall_near(graph_path: str, query: str, depth: int = 2,
                max_nodes: int = 300) -> List[Dict]:
    """Scoped recall: the neighborhood of `query`, not the whole closure. What the
    plugin calls per session. One call, terminates."""
    import sys as _sys
    _sys.path.insert(0, os.path.join(HERE, "dotcausal_package", "src"))
    from dotcausal import CausalReader
    # include_inferred=True: walk the concept bridges and inferred edges the graph
    # already carries — that is what lets recall hop from a query term to a needle
    # that shares no surface word with it, via a shared concept node.
    trips = CausalReader(graph_path, verify_integrity=False).get_all_triplets(
        include_inferred=True)
    # undirected adjacency: bridges connect peers, so follow edges BOTH ways
    adj: Dict[str, List[Dict]] = {}
    for t in trips:
        adj.setdefault(t["trigger"], []).append(t)
        adj.setdefault(t["outcome"], []).append(t)
    q = query.lower().strip()
    seen = {t["trigger"] for t in trips if q in t["trigger"].lower()} | \
           {t["outcome"] for t in trips if q in t["outcome"].lower()}
    frontier, hits = set(seen), []
    for _ in range(depth):
        nxt = set()
        for node in frontier:
            for e in adj.get(node, []):
                hits.append(e)
                # expand to BOTH endpoints — bridges are peer links, not directed
                for peer in (e["trigger"], e["outcome"]):
                    if peer not in seen and len(seen) < max_nodes:
                        nxt.add(peer); seen.add(peer)
        frontier = nxt
    return hits[:max_nodes]


def _project_of(path: str) -> str:
    """Project name from the .claude/projects/<slug>/ directory. The slug encodes
    the original path with '-' separators; the last meaningful segment after
    'Desktop'/'Documents'/'Projekte' is the project name."""
    parts = path.split(os.sep)
    try:
        i = parts.index("projects")
        slug = parts[i + 1].strip("-")
    except Exception:
        return "root"
    segs = [s for s in slug.split("-") if s]
    # drop the leading user-path prefix (Users, bhkmie, Desktop, Documents, ...)
    drop = {"Users", "bhkmie", "Desktop", "Documents", "Projekte", "Research"}
    meaningful = [s for s in segs if s not in drop]
    return ("-".join(meaningful[:2]) or "root")[:28]


def iter_sessions(root: str, min_bytes: int = 10240) -> Iterator[str]:
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(".jsonl"):
                p = os.path.join(dirpath, f)
                try:
                    if os.path.getsize(p) >= min_bytes:
                        yield p
                except OSError:
                    continue


def recall(causal_path: str, query: str, top: int = 12) -> str:
    """Read the persisted session-memory graph and answer a recall query against
    it — instantly, from the materialized graph, no re-reading transcripts. Two
    intents: a concept ("what touched X" → sessions mentioning it) or a project
    ("project Y" → its sessions and their intents)."""
    sys.path.insert(0, os.path.join(HERE, "dotcausal_package", "src"))
    from dotcausal import CausalReader
    trips = CausalReader(causal_path).get_all_triplets(include_inferred=False)
    q = query.lower().strip()
    intents = {t["trigger"]: t["outcome"] for t in trips if t["mechanism"] == "intent"}
    lines = []
    # concept recall: which sessions mention something matching the query
    hits = [t for t in trips if t["mechanism"] == "mentions" and q in t["outcome"].lower()]
    if hits:
        lines.append(f"sessions touching '{query}':")
        seen_s = set()
        for t in hits:
            s = t["trigger"]
            if s in seen_s:
                continue                         # one line per session
            seen_s.add(s)
            lines.append(f"  {s}  —  {intents.get(s, '(no recorded intent)')[:80]}")
            if len(seen_s) >= top:
                break
    # project recall: sessions under a matching project
    psess = [t for t in trips if t["mechanism"] == "has-session" and q in t["trigger"].lower()]
    if psess:
        lines.append(f"\nsessions in project '{query}':")
        for t in psess[:top]:
            s = t["outcome"]
            lines.append(f"  {s}  —  {intents.get(s, '')[:80]}")
    return "\n".join(lines) if lines else f"nothing in memory matches '{query}'."


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "recall":
        print(recall(sys.argv[2], " ".join(sys.argv[3:])))
