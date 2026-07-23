# F6 — Train short, deploy unbounded

**Statement (from `FOUNDATIONS.md` F6).** An operator with no absolute-position term (NoPE — the
only index-dependence is through lags) is in-distribution at every sequence length. Combined with
F2's exactness license, training at tiny horizons yields deployment at unbounded horizons. This is
one symmetry — shift-equivariance in time — measured on two independent axes: raw sequence
**length** (how far the position index runs) and **gap** (how long the input goes silent while a
specific keyed binding must survive). Same mechanism, same falsifier shape, two different attack
surfaces.

---

## Axis 1 — Length: train at T=32, hold flat to 4096× and beyond

From `results/scale_to_the_wall.json` (referenced in README §4), the NoPE-Selective model trained
at `T=32` and evaluated at the recurrent `O(1)` deployment path:

| eval length | extrapolation | PPL | ratio vs T=32 |
|---|---|---|---|
| 8,192 | 256× | 149.9 | ×0.92 |
| 32,768 | 1024× | 158.3 | ×0.97 |
| 65,536 | 2048× | 156.8 | ×0.96 |
| **131,072** | **4096×** | 160.8 | **×0.98** |

Flat to 4096× — the ratio does not drift monotonically with length, it oscillates in a narrow
band around 1.0. This holds with seed-robustness (n=5, seeds {1,7,42,123,2024}) at 256×:
Selective-NoPE **×0.93 ± 0.00** (std rounds to zero), vs Selective+PE **×7.05 ± 2.34** (breaks on
every seed) — `results/length_seed_robustness_d128.json`.

**Doubly `O(1)` — the corpus removed as a memory dependency too.** `results/scale_to_a_billion.json`:
the same `T=32`-trained model streamed **1,000,013,824 tokens** of C4 (31,250,432× the training
length) with the corpus itself entering only as a lazy iterator (no materialized document list).
Final PPL **247.5**, peak RSS **4.36 GB**. Across all 20 checkpoints (every 50M tokens):

| checkpoint | PPL | peak RSS |
|---|---|---|
| 100,000,000 tokens | 247.58 | 4.30 GB |
| 500,000,000 tokens | 247.54 | 4.35 GB |
| **1,000,013,824 tokens** | **247.5** | **4.36 GB** |

Two flat lines across a billion tokens: PPL moves **1.3 points total** (247.08–248.38, a 0.52%
band) and peak RSS moves **0.08 GB total** (4.28–4.36) over the entire run
(`checkpoints` array, `results/scale_to_a_billion.json`) — 153 minutes wall-clock, one machine
that never approached its 16 GB ceiling. The exactness guarantee underneath this number is F2's:
truncated-BPTT carry ≡ full-window BPTT at max-abs-delta **0.0**, grad-cosine **1.0**
(`results/streaming_check.json`) — the billion-token run is not an unbounded *approximation*, it
is the same operator evaluated for longer.

## Axis 2 — Gap: a movable knee from 32 to 2176/4096 tokens of silence

The length axis asks "how far can the position index run." The gap axis asks a structurally
different question: "how long can a *specific keyed binding*, written once, survive silence
before the read fails." This is `analysis/HOLO_STREAM_VERDICT.md`'s arc, and it is a knee, not a
slope, because F3's phase/magnitude split makes content-invariant-during-silence a structural
property — only the readout floor moves.

The knee arc, each step theory-led and each step landing on a committed result file:

| stage | mechanism | knee location | source |
|---|---|---|---|
| v1 (naive multi-chunk train) | — | collapses (P=1 pathology, gradient-blind write) | `results/holo_stream_recall.json` |
| v3 (train-short-eval-long) | cap training gap at single-chunk max | **32** (P=2 carried 0.79–0.83, P=1 at 1.00) | `results/holo_stream_recall_v3.json` |
| M2 (full-sequence training, curriculum bar 0.8) | raises effective γ via full-gradient write | **256** (`P30`-lineage seed cells: `knee_G=256`, `acc_at_ref=0.87–0.90`) | `results/holo_knee_bar08.json` |
| M3 (kickstart + magnitude-normalized read) | independently movable readout floor (F3) | **512**, seed-stable (`seed0_V2_kickstart_magnorm`, `seed1_V2_kickstart_magnorm` both `knee_G=512`) | `results/holo_magread.json` |
| P30 (eval-time phase-clamp + magnitude-refresh, jointly) | kills filler pollution (clamp) *and* feeds magnitude (refresh) — neither alone moves it | **mean 2176** (`mean_knee_G: 2176.0`, `clamp_refresh` arm), **individual seed ceiling 4096** (`seed0_clamp_refresh: knee_G=4096`, the top of the swept gap range) | `results/holo_clamp_refresh.json` |

The clamp-only and refresh-only ablations in the same file isolate the mechanism: `clamp_norefresh`
mean knee **640**, `unclamped_refresh` mean knee **512**, `unclamped_norefresh` (neither lever)
**768** — neither lever alone beats the baseline, only the **combination** reaches 2176.
`per_arm_summary` in `holo_clamp_refresh.json` shows why: `clamp_refresh` drives
`mean_drift_at_512` to **0.0** (phase pollution removed) while pairing it with `refresh: true`
(magnitude fed) — an interaction, not a sum, exactly as `HOLO_STREAM_VERDICT.md` states it. A
32→2176-mean (68×) knee shift in the same investigation, each step a registered prediction scored
against the data before the next step ran.

**The φ-drift falsifier locks the invariance claim underneath the whole arc.** `results/phi_drift.json`:
under a forced-zero drive, `|Δφ| ≈ 1e-8` (machine precision) across gaps up to 512, in both trained
and untrained models — content is written once and never evolves absent input. The knee is
therefore provably a **magnitude/readout** phenomenon, not silent content drift, which is what
licenses treating the readout floor (§ magnitude-normalized read, clamp, refresh) as the lever
instead of hunting for a phase-stability fix that was never needed.

---

## The honest limit of the principle: multi-chunk training is gradient-blind to the write

`analysis/HOLO_STREAM_VERDICT.md` (v1/v2 sections) is the falsification that makes the rest of
this document trustworthy. The *naive* application of F6 — just let the curriculum grow the gap
past a single chunk during training — does not work, and the reason is structural, not a tuning
miss:

- **v1**: P=1 carried collapses to chance (0.04–0.10). Diagnosis: the curriculum clears
  `acc > 0.9` every iteration and rockets the gap 2→128 within ~10 iterations; once the sequence
  spans multiple chunks, truncated BPTT gives the query-position loss **zero gradient path back to
  the write phase** (the KV write sits behind a detached chunk boundary). ~1990 iterations then
  train on a signal that structurally cannot teach writing, and the model unlearns.
- **v2 (patience curriculum)** was the registered fix — grow the gap only after sustained mastery
  — and it was **FALSIFIED**: P=1 collapses again (carried 0.045–0.085 ≈ chance) *even though* the
  curriculum this time demonstrably consolidated first (`Gcur` reached 26/40 via sustained >0.9
  phases). Patience delays the collapse; it does not prevent it, because the blindness is not
  about training *speed* — it is that the gradient literally cannot reach the write once the
  boundary is crossed. (P=4 cells are bit-identical v1↔v2 — patience never triggered there, an
  incidental determinism check.)

The fix that actually works is **v3, train-short-eval-long**: cap the *training* gap at the
single-chunk maximum and let all larger gaps be pure eval-time extrapolation — i.e., apply F6's
own recipe (train short, deploy long) to the gap axis instead of trying to train through it. This
is why the knee arc above starts at v3, not v1: the length-invariance symmetry only pays off once
you stop trying to gradient-train the thing it makes unnecessary to gradient-train. **F2's
train-anyhow/deploy-chunked decoupling is load-bearing here in a specific, falsifiable way**: it
does not mean any training layout works — the detached chunk boundary genuinely blocks gradient
to the write — it means the correct move is to train inside the boundary (single-chunk) and rely
on the *deployment*-side exactness (F2) plus the *length*-side invariance (F6) to carry the
learned behavior out to gaps the training loop never had gradient access to.

---

## What this chains into

One shift-equivariance property, two axes: sequence length holds flat to **4096×** in-distribution
(and the corpus-as-iterator removal pushes the *effective* horizon to a billion-plus tokens at
constant memory), and keyed-binding persistence through silence holds flat with a knee that moved
**32→2176 (mean) /4096 (seed ceiling)** — a 68×+ shift — by attacking the readout floor, not the
position index, because F3 already proved content survives silence exactly and only the magnitude
decays. The one honest wall found on either axis (multi-chunk gradient-blindness) is not a wall in
the *architecture* — it is a wall in one particular *training recipe*, and the repaired recipe
(v3) is itself an instance of the same train-short-deploy-long principle applied one level down.

## What is not yet measured

- The gap-axis knee arc (§ Axis 2) has not been run at the length-axis scale (billion-token
  regime) — the largest measured gap is 4096 tokens (one seed, `clamp_refresh`), not millions.
  Whether the same clamp+refresh combination holds a knee at gaps several orders of magnitude
  larger, the way the length axis holds flat to 4096× training length, is untested.
  `[not yet measured — candidate experiment: extend `holo_clamp_refresh` gap sweep past 4096,
  analogous to `scale_to_the_wall.py`'s length ladder]`
- `holo_clamp_refresh.json`'s own acceptance checks `P30c` (zeroed-at-gap null within 5pp of
  chance, every arm) is explicitly **NOT MET** — flagged here at full strength rather than
  smoothed over: the null-control discipline that holds cleanly elsewhere in the repo (e.g.
  `results/holo_heldout_keys.json`'s `beats_chance_3x: false` on the zeroed arm) needs a second
  look in this specific sweep — the ledger's diagnosis (PREDICTIONS.md P30 scoring): the 12/48
  out-of-band null cells are symmetric around chance with mean deviation +0.005, i.e. sampling
  noise at the small large-G eval batches (12–25 trials), not information manufacture; the
  eps-guard was verified in isolation. `[refinement candidate: re-run the null protocol with
  larger eval batches at G ≥ 1024]`
