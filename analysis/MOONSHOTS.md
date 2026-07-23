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
| MS3 | **Domain-shock resilience** (C4→code→C4) | gating+dosed replay = a natural anti-forgetting machine: continual learning at O(1) state without any CL machinery | POS, M4 ✅ | ✅ |
| MS4 | **Curiosity homeostasis** (self-set q) | the organism sets its own learning threshold; adaptive q beats every fixed q on non-stationary streams | MS3 harness | ⏳ |
| MS5 | **Language-stream holographic recall** (holo graft on the 400M-token POS model) | the knee/carrier story leaves synthetic MQAR: real facts recalled over real silence in a real language stream | M3 ✅, POS ckpt | ⏳ |
| MS6 | **CHIMERA — the complete organism** | all validated organs in ONE process on real text, measured against the sum of its parts | MS1, MS2/M4, MS3 | ⏳ |
| MS7 | **Two organisms, one index** (social memory) | organism B learns from A's surprises: collective memory across O(1) individuals | M5 ✅ | ⏳ |
| MS8 | **Hot-swap growth** (in-flight network surgery, d_model grows, state migrated, no restart) | the brain grows without sleeping; the twin experiment is the control | POS twin ✅ | ✅ |
| MS9 | **Auto-falsifier** (PREDICTIONS.md → machine-checkable scores) | the science machine audits itself: every new result JSON auto-scored against every registered prediction | — | 🔨 |
| MS10 | **The rent map of the phase** (P_max × d × n_slots phase diagram with critical line) | "where phase pays rent" becomes a law with a measured critical boundary, not an anecdote | M1/M6 ✅ | ⏳ |
| MS11 | **Weight hot-swap on the living stream** (old carried state × new weights, snapshot cross-matrix) | the state is a portable asset across model versions: session continuity through weight updates, measured | POS snapshots ✅ | ✅ |
| MS12 | **α-shut pollution control** (filler-write regularizer on the knee recipe) | the F3 pollution law made causal: a dial that moves the persistence knee past 1024 | F3 ✅, M3 ✅ | 💀 |
| MS13 | **The bit survives the surgery/swap** (beacon recall across widening + across weight updates) | the SLOW state is the portable asset: stored content survives model growth and model updates — or the state-code-drift law is discovered | MS8 ✅gate, MS11, carrier ✅ | ✅ |
| MS14 | **Adam-moment migration** (optimizer state through the duplication map) | growth becomes seamless: the transient vanishes (0.046→0.0035) | MS8 ✅ | ✅ |

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

## Launch queue (the pull system — standing order)

*Rule: every watch checks (a) harvests, (b) free run slots, (c) this queue.
If fewer than 2 reviewed harnesses are waiting, build the next entry NOW
(prediction registered first). Builders cost no CPU — an empty builder queue
next to free builders is the only true time-waste in this lab.*

1. MS7 — two organisms, one index (P31) — smoke running
2. MS4 — auto-q full (anti-windup fixed) — running
3. next builds on watch: MS10 rent map (born beast job), CHIMERA spec (lead)
2. MS14 — Adam-moment migration (P27) — full on beast
3. MS4 — auto-q curiosity homeostasis on the shock harness (P28) — smoke running
3. MS7 — two organisms, one index (P-tbd before build)
4. F2/F4/F6 unification docs (zero-CPU, any idle moment)
5. CHIMERA spec (MS6) — written by the lead, not delegated
6. MS10 — rent map of the phase (P-tbd before build)

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
4. Wave discipline: ≤4 RUNS in flight; builders are Sonnet subagents against
   exact specs; every build self-smokes; I review every line before a full
   run; controls before celebration (the lr-control lesson).
5. Pull system (day-3 addition): builder capacity is never idle — the launch
   queue above stays ≥2 deep in reviewed harnesses, refilled at every watch
   without being asked.

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
- day3 ~20:20 — MS8 ✅ + MS11 ✅ harvested; the TWO-TIMESCALE LAW emerges:
  the fast path forgets by design (P23a falsified: cold BEATS carried state;
  swap converges in 256 tokens — weight updates are a non-event, no lock-in),
  while the slow carrier holds the content (MS13 smoke: bit survives surgery
  AND swap at 1.000, cold collapses; carrier channel ADDRESS migrates
  Head0/Chan2→Head3/Chan8 across training yet the read stays perfect —
  portability by REDUNDANT ENCODING, not address stability). MS8: growth
  beats restart +0.127, surgery exact, no transient; capacity payoff not yet
  at 1.2M (deficit halved 0.073→0.036 — next lever: migrate Adam moments
  through the duplication map). **Coupling → MS6/CHIMERA: model updates and
  model growth are SAFE operations on the living organism; the state layer
  that needs care is exactly the layer the index+carrier own.**
- day3 ~20:30 — MS13 ✅ FULL: the bit survives BOTH operations at 1.000
  (surgery gate 2.4e-6; swap through gap 512; cold at chance). P26c honestly
  falsified-as-posed: the code is REDUNDANT (position+magnitude), permutation
  is not a sharp control here — recorded as the third outcome. Carrier
  address stable at convergence (H0C2==H0C2, gamma 1.000); smoke showed it
  CAN relocate with the read still perfect. The two-timescale law now has
  its content-level half measured. ms3 smoke complete: R3 (gated+sleep)
  forgets +0.136 = 28% of R1's +0.483 at near-R2 plasticity — P20's core
  ranking (R3<R2<R1) already visible; R3's phase-3 recovery (+0.123) is the
  full run's open question. ms3-full + MS12-full launched; ms5 smoke still
  computing its real-text evals (2h+, alive at 100% cpu — needs progress
  prints, builder feedback noted).
- day3 ~21:00 — FLEET ONLINE: beast (16 cores, torch 2.10) set up as the
  replication engine — running: M3 magnorm-knee seeds 2+3, F5 family-transfer
  seeds 43+44 (hardening 512-knee and P22 against seed luck). Found on intel:
  Davids WALLCLOCK lifetime run at 491M tokens, RSS flat 0.79GB, 5.7k tok/s —
  a second independent ~500M-token living stream on a SECOND architecture
  (linux/x86 vs macOS/ARM): cross-platform evidence for the streaming thesis.
  core is full (load 4.75/4) with 0 disk free — flagged, not touched. beast
  DNS was broken (dead tailscale MagicDNS + dead IPv6 upstream preferred);
  fixed via the official resolv.conf upstream-mode symlink, flagged to David.
  Queue note: MS10 (rent map, many cells) is the born beast job.
- day3 ~21:45 — MS12 💀 P25 FALSIFIED in-regime, and the forensic pass
  found something bigger: M3's curriculum NEVER ignited (final_train_gap=2
  in all six original cells) — the 512 knee was always kickstart+magnorm
  EXTRAPOLATION from a gap-2-trained model. The regularizer never had long
  gaps to train on; the lever cannot reach the mechanism in this regime.
  Sharper causal test registered (P29/MS12b): clamp α on fillers AT EVAL
  ONLY — if pollution binds the knee, the clamp must move it immediately,
  no training involved. Science hygiene note: 'bar 0.8 makes the curriculum
  ignite' in the M3 notes was a misreading; ignition never happened anywhere.
- day3 ~22:00 — P29 harvested: pollution is REAL and exactly removable
  (drift 0.0 under eval-clamp) but NOT the binding constraint past 512 —
  the filler write is a DOUBLE AGENT (pollutes phase, feeds magnitude:
  30-80x refresh unclamped vs 0.0007 collapse clamped). The knee is
  magnitude-bound. P30 registered: the 2x2 clamp x in-state magnitude
  refresh (the FOUNDATIONS-F3-disclosed variant) that fully disentangles
  the two axes — prediction: clamp+refresh moves the knee to >= 2048.
- day3 ~22:15 — P30 harvested, the knee arc is COMPLETE and mechanistically
  closed: 32 -> 256 (full-seq) -> 512 (magnorm read) -> 2176 mean / 4096
  end-of-range (eval clamp+refresh), every step registered before its data.
  The law: persistence = clean phase AND fed magnitude, and the two levers
  INTERACT (refresh-only is the worst arm — it rescales pollution; the
  combination is the jump). All three interventions are eval-time
  prostheses on a gap-2-trained model — folding them into training is a
  Phase-C question. Queue refilled per standing order: MS7 (P31) + F-docs.
- day3 ~22:50 — harvest wave scored+pushed: P20 partial (sleep = anti-
  forgetting organ CONFIRMED at 37% of full-gradient damage; recovery
  overdose discovered — phase-3 needs the dividend monitor), P27 (moment
  migration is a TRANSIENT KILLER 0.046->0.0035, not a capacity
  accelerator; growth-beats-restart now 3x replicated), 4-seed knee
  correction (the honest unit is the intervention effect, not the knee
  coordinate), P28 anti-windup found+fixed (auto matched fixed's
  forgetting at 18% fewer grad tokens even while saturated). Twin P5
  interim: the transient has the predicted SHAPE (monotone decay
  +0.007->+0.002 over 65min) at ~10x smaller magnitude than registered —
  MS11-coherent; final scoring in Phase B. Fleet note: beast family runs
  needed hard single-threading; C4-streaming jobs are the slow ones there.
