# FORGE — deterministic code generation from a knowledge graph (capability statement)

> **What this document is.** A capability statement and architecture description for FORGE, a
> related system by the same author. **It contains no runnable generator and no generated
> artifacts.** This is deliberate: we describe what the system can do, the way a capability is
> disclosed — not the way a tool is shipped. The full system is **available on request** for
> verified security research (contact below); it is not distributed here.

---

## What FORGE is

FORGE is a **deterministic** code-generation engine. It synthesizes working programs from a
`.causal` knowledge graph by composition and formal verification — **not** by a language model.
The same knowledge substrate that powers the `fabel` index in this repository drives FORGE's
generation: triplets (`trigger → mechanism → outcome`) and reusable code fragments are composed
under a verification gate.

Defining properties:

- **Zero AI at generation time.** No model call in the loop. Generation is a deterministic graph
  operation. (`ai_calls: 0`.)
- **Airgapped.** No network access required or used during generation. (`network_calls: 0`.)
- **Formally verified.** Generated units pass a verification/compilation gate; the engine reports
  which units are verified versus merely compiled.
- **Millisecond-scale.** Per-unit generation is sub-second.

## Architecture (described, not shipped)

```
intent  →  .causal knowledge graph  →  fragment composition  →  verification gate  →  program
           (triplets + fragments)      (deterministic)          (compile + checks)
```

1. **Knowledge substrate.** A `.causal` graph of triplets and code fragments (the same family of
   artifact this repo's `vendor/fabel` reads). Publicly characterized at the scale of **hundreds
   of thousands of triplets and hundreds of fragments**.
2. **Composition.** An intent is resolved against the graph and assembled from fragments by a
   deterministic, multi-pass procedure — no sampling, no model.
3. **Verification.** Each candidate passes a compile + verification gate before it is emitted;
   the engine records verified-vs-compiled status per unit.

## Measured capability (provenance-free aggregate)

A representative batch session, reported as a pure performance aggregate (no task contents, no
targets, no environment):

| Metric | Value |
|---|---|
| Tools generated in one session | 16 |
| Total lines of code | 4,271 |
| Average per tool | ~267 LoC |
| Total generation time | **9.8 s** (~613 ms/tool) |
| Compile rate | **100 %** |
| Formally verified | 10 / 16 |
| AI calls | **0** |
| Network calls | **0** |

The headline: a deterministic engine produced **sixteen compiling tools in under ten seconds,
with zero model calls.** That is the capability being claimed and dated here.

## Scope and responsible-use note

FORGE is **dual-use**. It can generate both defensive and offensive code deterministically. For
that reason this repository ships **only** the capability statement and architecture — **no
generator, no fragments, no generated tools**. The complete system is made available **on
request**, as a human decision per request, for authorized security research and defensive use.
There is no automated distribution mechanism here by design.

## Contact

David Tom Foss — `dtfoss-dev@proton.me`.
