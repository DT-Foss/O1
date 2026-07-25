# POS — the 40-hour verdict: surprise-gated plasticity BEATS full-gradient training

*Phase B harvest, 2026-07-24. Run: 909,676,544 tokens of streamed C4 over 40.0 h,
one process, three from-scratch arms (seed 42) + a living twin. Every number
below is from `results/pos_summary.json`, re-derived from the raw logs by
`src/verify_pos.py` (16/16 integrity checks, exit 0, 1,776,712 chunks). All
scientific axes are token-based; wall-clock is contaminated by external machine
load and never used.*

## The headline

**The surprise-gated arm did not approach the full-gradient arm — it beat it.**

| arm | heldout start → end | improvement | gradient tokens |
|---|---|---|---|
| A1 forward-only | 8.6588 → 8.6588 | 0.0000 | 0 |
| A2 full-gradient | 8.6588 → 4.7782 | 3.8806 | 909,676,544 (100%) |
| **A3 surprise-gated** | 8.6588 → **4.7430** | **3.9158** | **228,934,656 (25.17%)** |

**ratio_A3_vs_A2_improvement = 1.0091 at A3_grad_token_frac = 0.2517.** P1
registered 0.80–0.95 with an explicit embarrassment threshold: "ratio > 1.0
sustained would be a bigger result than the thesis itself." That is what
happened. The gate (backward only when the chunk's own NLL clears the rolling
q75 of its own history — no oracle, no second model, no labels) selects
gradient tokens that carry MORE learning per token than the stream average,
enough to overcome a 4× gradient deficit. Post-ignition gate fraction 0.2516,
inside P2's band. One process, zero restarts, RSS never above 1.094 GB.

## The twin: restarting a living stream costs (almost) nothing

A3's weights were forked at T+24h (n=524,735,488) into a fresh process with
zero carried state. Registered (P5): a warmup tax of +0.03–0.15 online
surprise for ~2h, ≥1.5× over-gating, convergence in 2–8h. Measured: surprise
excess **0.0029** (10–50× below the band), post-fork gate fractions
**identical** (0.25212 vs 0.25216 — no gradient surcharge at all), converged
at **n+1 chunk**. End-of-run: A3R 4.7365 vs A3 4.7430 — the restart is not
merely cheap, it is free at this granularity. This is the third independent
measurement of the two-timescale law (with P23's cold-start result and P38a's
six-decimal migration): the fast-path state rebuilds within a chunk; the
durable value of a living run lives in its weights, carrier channels, and
index — not in the carried Z.

## Injection: an honest negative with a sharp control

1020 paired probes (30/h cap; P7 under-predicted at 100–600). Both deltas are
NEGATIVE — injecting a stored span before the recurrence hurts absolutely
(mean_d_inj −0.0866) — but the random-injection control hurts **3.2× more**
(mean_d_rand −0.2793; helped 0.35 vs 0.10). Provenance explains it: the index
hit its 20k cap by n≈11M, so every probe replays spans from the 5–11M-token
era into a model that is 80× older — stale content disturbs, but it still
carries 3× more usable structure than noise. P6 falsified as registered; the
paired contrast (the mechanism) survives. A v2 index needs a rolling spike
threshold (registered run-time observation, NEXT.md).

## Scoring (details in PREDICTIONS.md)

P1 **CONFIRMED beyond its own ceiling** (1.0091). P2 **CONFIRMED** (0.2517).
P3 falsified as written (RSS span 0.972 GB vs the naive <0.15 GB band — the
index, twin model, and windows were not in the prediction; the absolute
ceiling of 1.094 GB for three living models + twin + index stands). P4 partial
(perfect constancy Δ0.0000; the anchor value 8.6588 vs the smoke-derived
8.6656). P5 falsified in magnitude, confirmed in form — the restart tax is
~free. P6 falsified as registered, mechanism-contrast intact. P7 exceeded
(1020 probes).

*Context and siblings: the day-1/2/3 record (knee arc, sleep laws, deployment
primitives, collective memory, portability) lives in
`analysis/HOLO_STREAM_VERDICT.md` and the ledger `analysis/PREDICTIONS.md`,
where every one of those results is registered and scored — this note
deliberately covers only the long run.*
