# causal_pipeline

**Text → knowledge graph → conversation. Deterministic, no LLM, no MLX.**

A **fully self-contained** pipeline that turns plain text into a `.causal`
knowledge graph and lets you talk to it — every stage rule-based and
inspectable, every answer traceable to a source. This is the LLM-free
replacement for the old MLX few-shot extractor: causal connectives are
*measured* structure in text, so a finite pattern set finds them, no model
required.

Nothing here points outside this folder. Code, data (`data/corpora/`), the
bundled `dotcausal` package, the language modules, and the graphs all live in
the repo — copy it anywhere and it runs. (FORGE is an optional, deferred
integration — see `TODO.md`.)

## Layout

```
extract/        rule + semantic extractors, Foss gate          (text → triplets)
build/          db → .causal, embedded inference                (triplets → graph)
language/       hsslm_s sentence-form modules + speakers        (graph → speech)
dotcausal_package/  the canonical .causal format (bundled)
data/corpora/   source text (faraday, base knowledge, ...)
graphs/         built .causal graphs + the FORGE snapshot
fabel.py        single-graph conversation
brain.py        multi-graph brain: mount / ingest / character
federation.py   federated graphs (isolation + bridging)
character.py    identity + decaying memory
forge_adapter.py  optional FORGE code-knowledge transcoder
```

```
 text ─▶ extract/ ─▶ SQLite DB ─▶ build/ ─▶ .causal ─▶ fabel.py ─▶ conversation
        rule_extractor          build_causal_from_db   (dotcausal)
        + Foss gate              + embedded inference
```

## The brain (multi-graph)

`brain.py` is fabel with **many graphs at once**. A base `.causal` is always
loaded (general speaking); domain graphs mount on demand into one shared entity
space, so a causal chain can cross module boundaries — load two graphs and the
brain answers a question neither could alone:

```
brain > :ingest smoking_facts.txt as smoking
brain > :ingest respiratory.txt as resp
brain > how does smoking lead to breathlessness?
        Smoking causes lung damage. Now, lung damage causes breathlessness.
          [0.70 explicit @smoking | smoking_facts.txt]  smoking -> lung damage
          [0.70 explicit @resp    | respiratory.txt ]  lung damage -> breathlessness
```

Neither graph has both endpoints; the brain bridges them through the shared
entity `lung damage`, and provenance shows which module each fact came from.

REPL: `:mount PATH [as NAME]`, `:unmount NAME`, `:mounted`, `:ingest SOURCE
[as NAME]` (file / folder / URL → extract → build → mount), `:topics`,
`:save PATH` (flatten the whole brain to one `.causal`).

```bash
python3 brain.py [base.causal]
```

FORGE-style code/repo → triplet scraping plugs in at `brain.py`'s
`SOURCE_ADAPTERS` seam — same `.causal` artifact, no LLM.

## Quick start

```bash
./run_pipeline.sh path/to/corpus.txt pharma     # extract → build
python3 fabel.py graphs/corpus.causal           # talk to it
```

Or step by step:

```bash
python3 extract/extract_to_db.py corpus.txt --db graphs/c.db --domain physics
python3 build/build_causal_from_db.py --db graphs/c.db -o graphs/c.causal
python3 fabel.py graphs/c.causal
```

## What each part does

| Part | Role |
|------|------|
| `extract/rule_extractor.py` | Deterministic triplet extraction. Causal connectives (causes / leads to / reduces / due to / …) split each sentence into trigger–mechanism–outcome. Entities are trimmed to the noun phrase nearest the connective; quantifications (40%, p<0.05, …) are kept verbatim. |
| `extract/extract_to_db.py` | Drives the extractor over files/dirs, passes every triplet through the 14-step **Foss validation gate**, writes survivors to a SQLite DB. |
| `extract/extract_causal_triplets_v2.py` | The pipeline's validation gate (`validate_triplet_v2`), bundled. Only the gate is used — no MLX. |
| `build/build_causal_from_db.py` | SQLite → `.causal`, runs the embedded 3-pass inference to amplify (transitive chains the text never stated). |
| `build/causal_format.py` | Local shim re-exporting the bundled `dotcausal` package. |
| `dotcausal_package/` | The canonical `.causal` format package (reader/writer/inference). |
| `fabel.py` | Conversational layer. Ask `what causes X`, `what does X cause`, `how does X lead to Y`, `tell me about X`. Answers carry confidence + source; unknown entities are refused, not invented. |
| `bank/faraday_bank.json` | Mined sentence-form patterns (optional; fabel falls back to plain phrasing without it). |
| `graphs/` | Built DBs and `.causal` files. |

## Why it's not "AI"

No weights, no inference-time model, no hallucination surface. The extractor
finds causal structure that is literally written in the text; the graph stores
it with provenance; `fabel` only ever speaks facts it can trace. Ask it
something the graph doesn't know and it says so — that refusal is the point.

## Trade-off to know

Tighter entity trimming (good for readability) reduces exact-match overlap
between entities, which lowers transitive inference amplification on prose-heavy
corpora. Dense, declarative text (one causal claim per short sentence) gives
both clean entities and high amplification. The `graphs/smoking_demo.causal`
graph shows the ideal case.

## Self-contained

Nothing here points outside this folder except the optional sentence-form bank
(`hsslm_s`). Copy the directory anywhere and it runs.
