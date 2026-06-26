# TODO

## Eval harness (substrate/eval/) — built + measured 2026-06-12

`eval_extractor.py` (P/R/F1 vs gold), `eval_loop.py` (repair convergence, mock
FORGE). Dashboard renders both reports.

Progress on the causal extractor, measured:
- start: P=0.69 R=0.56 F1=0.62
- fix over-segmentation (cut outcome at by/which/comma): F1=0.83
- fix verb regex bug (`suppresses?` never matched "suppress") + add verbs +
  sentence-initial because/since: F1=0.94
- strip comma appositives ("X, often due to stress, worsens Y"): F1=1.00 on the
  28-sentence gold set.

**Verb coverage was the bottleneck — FIXED with WordNet + spaCy (2026-06-12).**
- `build_verb_lexicon.py` harvests ~1150 causal verbs from WordNet ONCE into a
  static `causal_verbs.json` (no NLTK at runtime — just a JSON load). Tagged by
  polarity (increase/decrease/cause/damage/improve).
- Explicit CONNECTIVES still win first (exact mechanism wording); the lexicon is
  the fallback for the long tail. Deterministic suffix lemmatizer (incl. doubled
  consonant: spurred→spur) maps inflections to lexicon entries.
- Two guards keep the zero-false-positive property despite 1150 verbs:
  transitivity (reject verb+preposition / verb+-ly adverb: "opens at nine",
  "burned brightly") and OPTIONAL spaCy POS (only tokens tagged VERB qualify, so
  "sea levels threaten X" picks threaten not the noun levels). spaCy is
  lazy/optional — absent → surface heuristics, no hard dep.

Held-out gold set added (`causal_heldout.json`, never tuned against). Honest
generalization numbers, eval reports both:
- DEV:      P=1.00 R=1.00 F1=1.00 neg-acc=1.00 (overfit to the dev set)
- HELD-OUT: P=1.00 R=0.92 F1=0.96 neg-acc=1.00 (the real number)
The one held-out miss ("Fertilizer runoff pollutes rivers") is a spaCy POS error
(it tagged "runoff" as the verb) — inherent small-model limit, not worth
over-tuning. 0 false positives throughout.
- [ ] Optional: bigger spaCy model (md/lg) would fix the rare POS mistag, at the
      cost of a larger dependency. Current small model is the right tradeoff.

## FORGE integration (deferred — no live dependency)

FORGE (the code/repo → triplet engine) is **optional and not wired as a hard
dependency**. The repo runs fully without it.

What exists now:
- `forge_adapter.py` transcodes FORGE's `.causal` format (6-byte `CAUSAL`
  magic + zlib-msgpack) into the canonical format the brain reads.
- `graphs/forge.causal` — a snapshot of FORGE's code knowledge already
  transcoded (47,978 triplets), so `:forge` mounts it with no FORGE present.
- `:forge` in the brain mounts the cached snapshot; `:forge rebuild` re-runs
  the transcode from a FORGE knowledge dir.

To re-point at a live FORGE knowledge base later:
- set `FORGE_KB=/path/to/forge/knowledge`, or
- copy a FORGE `knowledge/` dir into `./forge_kb/`, then `:forge rebuild`.

Open items when we pick this back up:
- [ ] Inference over the full 48K-triplet FORGE KB takes ~314s (O(n²) fuzzy).
      Decide: materialize once and ship the inferred graph, or keep
      `include_inferred=False` at mount (current behavior).
- [ ] FORGE's `forge_query_kb` MCP returns empty triplet fields (built for
      codegen, not export) — that's why we dock to the `.causal` files
      directly. If FORGE adds a triplet-export API, prefer it.
- [ ] Decide whether FORGE code triplets belong in the general brain or as a
      separate "code brain" instance (homonyms: `socket`, `cell`, `key`...).

## The loop (PLAN → BUILD → TEST → LEARN) — prototype works

`loop.py` closes the fabel↔FORGE loop: fabel PLANs (graph + provenance), FORGE
BUILDs+TESTs (MCP generate/verify/execute, injected by the runtime — no hard
dep), and the outcome is written BACK as triplets (LEARN). Verified end to end:
"create a TCP port scanner" → 4 plan steps → FORGE 109 lines, proven → 3
learned triplets ("is built by / is verified by / runs successfully").

Open items:
- [ ] PLAN is thin — intent seed words match too few entities. Needs better
      intent→entity resolution (synonyms, multi-word, the semantic graph).
- [ ] LEARN writes to a separate db; wire it to rebuild+remount so growth is
      live within a session, and decide provenance domain handling.
- [x] Self-repair direction (`repair_loop.py`): TEST fails → fabel DIAGNOSEs
      the error into a refined intent → BUILD again, up to max_rounds. fabel
      is the language organ on BOTH sides (UNDERSTAND in, VERBALIZE out).
      Verified: NameError → "(define X before using it)" → fixed in round 2.
- [ ] **Functional correctness gap**: FORGE's `execute` returns success=true
      when code RUNS, not when it does what the intent MEANT. Real example:
      "reverse a string" → FORGE emitted `text.title()`, ran fine, wrong job.
      The loop currently trusts "ran ✓". Need a semantic check (does output
      match intent?) before LEARN marks it a success — otherwise the graph
      learns false "this works" triplets.
- [ ] Multi-step plans: chain several FORGE builds for a compound intent.

## Other

- [ ] Entity normalization between extraction and build, so prose-extracted
      variants ("cortisol levels" vs "elevated cortisol") chain for inference.
- [ ] Contradiction detection across modules (A→promotes→B vs A→prevents→B).
- [ ] "Why do you think that?" intent exposing provenance as a follow-up.
