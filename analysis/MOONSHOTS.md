# MOONSHOTS — the living portfolio (dynamic, self-reprioritizing)

*Opened day 1 ~15:00, mandate: 10 moonshots, four levels up, autonomous,
each result reprioritizes the rest. Status legend: ⏳ queued · 🔨 building ·
🚀 running · ✅ harvested · 💀 killed-by-evidence. Every launch registers a
prediction in PREDICTIONS.md first. POS long run always has priority;
everything here runs nice-19.*

## The end-state these aim at

Not "better recall numbers" — **the complete organism**: one process that
reads forever, chooses when to learn (POS ✅), carries keyed memory through
silence (knee arc ✅), consolidates its own selected memories in dosed sleep
(M4 ✅), consults an external index in flight (M5 ✅) — and beyond: learns to
use its consultations, dreams, survives domain shocks, regulates its own
curiosity, grows without restarting, and shares memory with others.

## Portfolio

| # | Moonshot | four-levels-up claim | deps | status |
|---|---|---|---|---|
| MS1 | **Learned to be reminded** (consultation IN training) | the state+index seam disappears: hybrid ≥0.85 at P=16, and the model learns to ARBITRATE state vs. injection | M5 ✅ | 🔨 |
| MS2 | **The dream generator** (generative replay) | sleep without storage: training on the model's OWN sampled dreams rivals stored-span replay — memory becomes optional | M4 ✅ | 🔨 |
| MS3 | **Domain-shock resilience** (C4→code→C4) | gating+dosed replay = a natural anti-forgetting machine: continual learning at O(1) state without any CL machinery | POS, M4 ✅ | 🔨 |
| MS4 | **Curiosity homeostasis** (self-set q) | the organism sets its own learning threshold; adaptive q beats every fixed q on non-stationary streams | MS3 harness | ⏳ |
| MS5 | **Language-stream holographic recall** (holo graft on the 400M-token POS model) | the knee/carrier story leaves synthetic MQAR: real facts recalled over real silence in a real language stream | M3 ✅, POS ckpt | ⏳ |
| MS6 | **CHIMERA — the complete organism** | all validated organs in ONE process on real text, measured against the sum of its parts | MS1, MS2/M4, MS3 | ⏳ |
| MS7 | **Two organisms, one index** (social memory) | organism B learns from A's surprises: collective memory across O(1) individuals | M5 ✅ | ⏳ |
| MS8 | **Hot-swap growth** (in-flight network surgery, d_model grows, state migrated, no restart) | the brain grows without sleeping; the twin experiment is the control | POS twin ✅ | ⏳ |
| MS9 | **Auto-falsifier** (PREDICTIONS.md → machine-checkable scores) | the science machine audits itself: every new result JSON auto-scored against every registered prediction | — | 🔨 |
| MS10 | **The rent map of the phase** (P_max × d × n_slots phase diagram with critical line) | "where phase pays rent" becomes a law with a measured critical boundary, not an anecdote | M1/M6 ✅ | ⏳ |

## Dynamics rules (how the portfolio reprioritizes itself)

1. **Every harvest triggers a portfolio review** — written as a dated line
   under "Reprioritization log" below, committed.
2. Known couplings, pre-declared: MS2 ≥ stored-replay ⇒ MS6 drops the span
   store (dreams replace it) and MS7 shares dreams, not spans. MS1 breaks
   the 0.84 ceiling ⇒ MS6's index path is load-bearing; MS1 fails ⇒ MS6
   ships state-only and MS7 is deprioritized. MS3 shows replay prevents
   forgetting ⇒ MS6's headline metric becomes continual, not stationary.
   MS4 beats fixed q ⇒ the Phase-C long run adopts auto-q.
3. A moonshot is 💀 the moment evidence kills its premise — killed is a
   result, logged with the number that killed it.
4. Wave discipline: ≤4 in flight; builders are Sonnet subagents against
   exact specs; every build self-smokes; I review every line before a full
   run; controls before celebration (the lr-control lesson).

## Reprioritization log

- day1 15:00 — portfolio opened; wave 3 = MS1+MS2+MS3+MS9 (MS4 waits for
  MS3's non-stationary harness; MS5/MS6 wait for wave-3 signals).
