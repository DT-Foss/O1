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
| MS1 | **Learned to be reminded** (consultation IN training) | the state+index seam disappears: hybrid ≥0.85 at P=16, and the model learns to ARBITRATE state vs. injection | M5 ✅ | ✅ |
| MS2 | **The dream generator** (generative replay) | sleep without storage: training on the model's OWN sampled dreams rivals stored-span replay — memory becomes optional | M4 ✅ | 💀 |
| MS3 | **Domain-shock resilience** (C4→code→C4) | gating+dosed replay = a natural anti-forgetting machine: continual learning at O(1) state without any CL machinery | POS, M4 ✅ | 🔨 |
| MS4 | **Curiosity homeostasis** (self-set q) | the organism sets its own learning threshold; adaptive q beats every fixed q on non-stationary streams | MS3 harness | ⏳ |
| MS5 | **Language-stream holographic recall** (holo graft on the 400M-token POS model) | the knee/carrier story leaves synthetic MQAR: real facts recalled over real silence in a real language stream | M3 ✅, POS ckpt | ⏳ |
| MS6 | **CHIMERA — the complete organism** | all validated organs in ONE process on real text, measured against the sum of its parts | MS1, MS2/M4, MS3 | ⏳ |
| MS7 | **Two organisms, one index** (social memory) | organism B learns from A's surprises: collective memory across O(1) individuals | M5 ✅ | ⏳ |
| MS8 | **Hot-swap growth** (in-flight network surgery, d_model grows, state migrated, no restart) | the brain grows without sleeping; the twin experiment is the control | POS twin ✅ | ⏳ |
| MS9 | **Auto-falsifier** (PREDICTIONS.md → machine-checkable scores) | the science machine audits itself: every new result JSON auto-scored against every registered prediction | — | 🔨 |
| MS10 | **The rent map of the phase** (P_max × d × n_slots phase diagram with critical line) | "where phase pays rent" becomes a law with a measured critical boundary, not an anecdote | M1/M6 ✅ | ⏳ |
| MS11 | **Weight hot-swap on the living stream** (old carried state × new weights, snapshot cross-matrix) | the state is a portable asset across model versions: session continuity through weight updates, measured | POS snapshots ✅ | 🔨 |
| MS12 | **α-shut pollution control** (filler-write regularizer on the knee recipe) | the F3 pollution law made causal: a dial that moves the persistence knee past 1024 | F3 ✅, M3 ✅ | 🔨 |

## Foundations track (F1–F6) — claim the primitives, not the effects

*Opened day 2 ~19:15 after the o1-state push. Every foundation gets: one
umbrella formulation in FOUNDATIONS.md (maximally general variant language,
pushed = defensively published breadth) + where cheap, one locking experiment.*

| # | Foundation | locking move | status |
|---|---|---|---|
| F1 | **The surprise calculus** — own NLL as the universal control signal (when to learn / what to store / when to consult / when to sleep / how curious) | CHIMERA (MS6) is the proof | ⏳ |
| F2 | **The exactness license** — bounded contraction ⇒ detach-carry exact ⇒ train-anyhow/deploy-chunked equivalence | formalize + equivalence sweep table (numbers exist for 3 operators) | ⏳ |
| F3 | **Phase–magnitude separation** — phase=content (written, never evolved), magnitude=persistence (learnable γ) | direct φ-drift measurement across gaps (theory falsifier 1, never run) | ✅ |
| F4 | **The two-system law** — sharp capacity cliff in gated readouts ⇒ compounding belongs to the index | unification doc (cliff + rank-1 theorem + M5 hybrid) | ⏳ |
| F5 | **Operating modes are family-generic** — POS gating + sleep run on the Mamba/S6 configuration of the same codebase | POS-on-Mamba head-to-head (doubles as the DD baseline) | ✅ |
| F6 | **Train-short, deploy-unbounded** — shift-equivariance (length) and gap extrapolation are one principle on two axes | unification doc | ⏳ |

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
- day1 ~17:10 — MS9 ✅ (auto-falsifier live; side gift: live interim_ratio
  1.002 @410M). MS1 ✅ harvested: the READ ceiling is broken and hardened
  (0.99–1.00 at P=2, both seeds, both G — vs M5's 0.72–0.84); hybrid@P16
  stays ~0.49 (state interference, not reading, is the high-P bottleneck);
  the conflict variant exposed trust-decay-not-arbitration (hybrid falls WITH
  random rising). **Coupling applied → MS6 (CHIMERA): index consultation =
  in-sequence reminders trained with reliable-index statistics (base setup);
  arbitration-by-contrastive-signal is its own later moonshot, NOT folded
  into MS6.**
- day1 ~17:45 — MS2 💀 (killed by evidence, the good kind): P19 FALSIFIED —
  sleep > fresh > dream > dream_shuffled; dreams lose even to fresh data
  (self-sampling is self-confirmation: generator's own dreams NLL 4.71 vs
  world 4.77). **Coupling applied → MS6/MS7 KEEP the span store.** Deeper
  find: the stored spans are no longer surprising to today's snapshot
  (4.57 < dreams) yet replaying them still wins — the sleep dividend is
  CONSOLIDATION OF THE LEARNED, not relearning of the hard; and it shrinks
  as the snapshot matures (+0.033 → +0.0002 over ~250M tokens). MS6's sleep
  organ should therefore replay RECENT spans (freshness-weighted), and the
  dividend is largest mid-training — both now design inputs for CHIMERA.
- day1 ~18:10 — M4/P15 final: FAIL in registered form at the mature snapshot
  (~430M: gap −0.006, dividends decay +0.023→+0.004→−0.042 with healthy span
  pools) — combined with MS2's finding, the sleep dividend has a LIFE CURVE:
  strong early (+0.033 @10M), positive mid (@146M smoke), spent late.
  **Coupling → MS6: the sleep organ gets a dividend monitor (measure each
  sleep's heldout delta, throttle sleep when ≤0) instead of a fixed cadence.**
- day2 ~19:15 — o1-state pushed (full history, worldwide burn, grace clock
  running). Foundations track opened; F3+F5 builders launched; FOUNDATIONS.md
  (the $0 breadth publication) + README rebrand queued for tonight. New
  standing rule: every harvest is pushed immediately (David's mandate).
- day2 ~19:30 — F3 ✅ LAW LOCKED (a=0 ⇒ |Δφ|≈1e-8) + discovery: real fillers
  actively pollute the phase (α>0, magnitude×35) — the knee's far-field
  falloff is pollution, not decay; α-shut-on-fillers is the next knee lever.
- day2 ~19:45 — F5 ✅ P22 CONFIRMED both parts at 6M tokens: POS-ratio
  ratio-of-ratios S6/GSSM = 0.98 (0.9337 vs 0.9523, gates 23%/23%) — the
  gating benefit belongs to the operator FAMILY, not the house architecture;
  and the DD baseline lands as a lead: GSSM-full beats S6-full by 0.156 nats
  at scan-param parity 1.0016, same stream/seed/pipeline. **Coupling applied →
  CHIMERA (MS6) claims run family-wide by construction; a surprise-gated
  Mamba is an instance of the disclosed system (FOUNDATIONS.md F5 now has its
  locking experiment).**
- day3 ~20:00 — wave 5 opened (the deployment primitives, FTO-led): MS11
  weight-hot-swap (P23, near-free via the snapshot archive), MS8 hot-swap
  growth reframed honestly (Net2Net widening is 2015 prior art; OUR claim is
  exact carried-Z migration on a live stream with no transient) (P24), MS12
  α-shut (P25, the causal intervention on F3's pollution law). MS7 shared
  index queued as wave 6. Predictions committed BEFORE builders launched —
  the P21/P22 caveat does not repeat.
- day3 ~20:15 — wave-5 harnesses built (all three self-smoked): MS8 smoke
  shows exact surgery (5e-6), no transient, growth-beats-restart (+0.09);
  capacity payoff needs full scale. MS11 smoke PRE-FINDING: cold start is NOT
  worse than the carried state on the plain-LM path — which is F2 from the
  other side (receptive field ~5-8 tok ⇒ fast-path Z rebuilds within a chunk;
  shuffled-control +0.9 excess shows structure still matters). If the full
  run confirms, P23(a) is honestly falsified and the REAL portability claim
  moves to the slow channels (carrier/index) — sharper follow-up: beacon
  recall across a weight swap. MS12 smoke: regularizer mechanically works
  (alpha_filler drops dose-dependently); causal knee test needs full budget
  (curriculum must actually reach long gaps first).
