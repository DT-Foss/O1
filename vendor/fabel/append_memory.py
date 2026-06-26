"""
append_memory.py — ranklatschen: add ONE finished session to the persisted memory
graph, in place, without recomputing anything.

Why this is cheap (and why the big bake ran in 1s): the .causal "inference" for
session memory is NOT iterative causal closure — it is grouping co-mentions into
session<->session bridges. That is O(edges), a single pass, no matrix work. Appending
one session only builds bridges for ITS topics against sessions already in the graph,
then writes the file back with all inferred edges still materialized (so the next
open stays instant).

Usage (what Codex calls at turn-end):
    python3 append_memory.py <graph.causal> <session.jsonl> [claude|codex]
Prints a one-line metrics record (JSON) to stdout for logging.
"""
from __future__ import annotations
import json, os, sys, time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "dotcausal_package", "src"))
from dotcausal import CausalReader, CausalWriter
from session_memory import extract_session, detect_tool

HUB_CEIL = 40          # topics in more sessions than this are noise hubs — no bridge


def append_session(graph_path: str, session_jsonl: str, tool: str = "") -> dict:
    t0 = time.time()
    tool = tool or detect_tool(session_jsonl)
    existing = CausalReader(graph_path).get_all_triplets(include_inferred=True)

    # current topic -> sessions index (for bridging the new session in)
    topic2sess = defaultdict(set)
    seen_keys = set()
    for t in existing:
        seen_keys.add((t["trigger"], t.get("mechanism", ""), t["outcome"]))
        if t["mechanism"] == "mentions":
            topic2sess[t["outcome"]].add(t["trigger"])

    new_edges = extract_session(session_jsonl, tool=tool)
    if not new_edges:
        return {"appended": 0, "bridges": 0, "skipped": "empty session"}

    sid = new_edges[0]["a"] if new_edges else "?"
    bridges = []
    for e in new_edges:
        if e["mech"] == "mentions":
            topic = e["b"]
            others = topic2sess.get(topic, set())
            if 0 < len(others) <= HUB_CEIL:
                for o in sorted(others):
                    bridges.append((e["a"], o, topic))
            topic2sess[topic].add(e["a"])

    # write back: everything that was there + new explicit + new bridges
    w = CausalWriter()
    for t in existing:
        w.add_triplet(trigger=t["trigger"], mechanism=t.get("mechanism", ""),
                      outcome=t["outcome"], confidence=t.get("confidence", 1.0),
                      is_inferred=bool(t.get("is_inferred")))
    added = 0
    for e in new_edges:
        k = (e["a"], e["mech"], e["b"])
        if k in seen_keys:
            continue
        w.add_triplet(trigger=e["a"], mechanism=e["mech"], outcome=e["b"],
                      confidence=e.get("conf", 1.0), is_inferred=False)
        added += 1
    for a, b, topic in bridges:
        w.add_triplet(trigger=a, mechanism=f"shares-topic:{topic}", outcome=b,
                      confidence=0.7, is_inferred=True)
    w.save(graph_path)

    # metric: tokens this session would cost to RE-READ raw vs the graph slice
    raw_tok = os.path.getsize(session_jsonl) // 4
    return {"session": sid, "tool": tool, "appended": added,
            "bridges": len(bridges), "seconds": round(time.time() - t0, 2),
            "raw_tokens_if_reread": raw_tok,
            "graph_now_edges": len(existing) + added + len(bridges)}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: append_memory.py <graph.causal> <session.jsonl> [tool]")
        sys.exit(1)
    rec = append_session(sys.argv[1], sys.argv[2],
                         sys.argv[3] if len(sys.argv) > 3 else "")
    print(json.dumps(rec))
