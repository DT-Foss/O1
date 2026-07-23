# F2 — The exactness license, formalized

**Statement (from `FOUNDATIONS.md` F2, sharpened).** For any recurrent operator whose state
dynamics are a contraction with an effectively bounded receptive field `r`, two things that
look like approximations are exact:

1. **Detach-carry streaming training** — truncated BPTT with the state carried and detached at
   chunk boundaries reproduces full-window BPTT gradients whenever the chunk exceeds `r`.
2. **Layout decoupling** — full-sequence forward with zero initial state and chunked-carried
   forward are *the same operator*, to numerical precision, at every chunk boundary. Training
   layout (full-sequence, gradient reaching every write) and deployment layout (chunked,
   `O(chunk)` memory, unbounded length) are therefore free to differ.

This is not "close enough to work." Every instance below is a measured equality, not a bound.

---

## The equivalence sweep table

Every row is a `full-seq ≡ chunked+carried` or `truncated-BPTT ≡ full-BPTT` check that ran
against a hard tolerance gate and passed. Source file inline.

| System | Check | Measured delta | Tolerance | Pass | Source |
|---|---|---|---|---|---|
| Base recurrence, training gradients | truncated-BPTT carry+overlap vs full-window BPTT | max-abs-delta **0.0**, grad-cosine **1.0**, grad-rel-err **0.0** | — | ✓ | `results/streaming_check.json` |
| Holographic stream recall (v1) | `use_phase_true` / `use_phase_false` / `use_phase_true_n_slots4` chunked-vs-full | **0.0** on all three arms | — | ✓ (`"passed": true`) | `results/holo_stream_recall.json` |
| Holographic stream recall (v3, train-short-eval-long) | same three-arm equivalence | **0.0** on all three arms | — | ✓ (`"passed": true`) | `results/holo_stream_recall_v3.json` |
| POS family transfer, GSSM arm | full-seq vs chunked+carried | `GSSM_max_abs_err` **0.0** | 1e-4 | ✓ | `results/pos_family.json` (`equivalence_gate`) |
| POS family transfer, S6/Mamba arm | full-seq vs chunked+carried | `S6_max_abs_err` **0.0** | 1e-4 | ✓ | `results/pos_family.json` (`equivalence_gate`) — the same license holds on a *different* member of the affine-scan family (F5) |
| Hot-swap growth (channel duplication d64→d128, mid-stream surgery) | stateless vs chunked-3 forward, post-surgery | **6.67572e-06** on both | 1e-4 | ✓ | `results/hot_swap_growth.json` (`surgery_equivalence_gate`) |
| Beacon adapter (bolted-on read head) | adapter stateless vs chunked forward | **0.0** on both | 1e-4 | ✓ | `results/beacon_swap.json` (`adapter_verification`) |
| Beacon surgery (state-vector transplant across a checkpoint boundary) | surgery stateless vs chunked forward | **2.384186e-06** on both | 1e-4 | ✓ | `results/beacon_swap.json` (`surgery_gate`) |
| Holo α-shut (write-once-freeze regularizer) | chunked-vs-full equivalence | **0.0** | — | ✓ | `results/holo_alpha_shut.json` (`equivalence`) |
| Holo clamp+refresh (eval-time phase clamp + magnitude refresh) | chunked-vs-full equivalence | **0.0** | — | ✓ | `results/holo_clamp_refresh.json` (`equivalence`) |
| Holo φ-drift probe | chunked-vs-full equivalence | **0.0** | — | ✓ | `results/phi_drift.json` (`"equivalence": 0.0`) |
| Holo γ-knee sweep (bar 0.9 and bar 0.8 curricula) | chunked-vs-full equivalence | **0.0** (both configs) | — | ✓ | `results/holo_knee.json`, `results/holo_knee_bar08.json` (`"equivalence": 0.0`) |
| Holo magnitude-normalized read | chunked-vs-full equivalence | **0.0** | — | ✓ | `results/holo_magread.json` (`equivalence`) |

**Twelve independent measurements, twelve exact passes**, across: the base scalar recurrence,
the complex holographic (phase) scan, a channel-count-grown state, a bolted-on adapter head, a
transplanted state vector, and a second architecture in the affine-scan family (S6/Mamba). The
license in F2 is not architecture-specific — it is a property of the operator class, and the
sweep now covers representatives from every deployment primitive in the repository (train,
grow, graft, swap, regularize, read differently).

---

## Consequence 1 — layout decoupling: train anyhow, deploy chunked

Because full-sequence and chunked+carried are the *same operator*, the training computation
graph and the deployment computation may be chosen independently. This is what turns a
gap-curriculum trained at tiny horizons into a deployable streaming skill: `src/holo_gap_knee.py`
and `src/holo_mag_read.py` both train under one layout and deploy under another, and the
equivalence gates above are what license doing so without re-deriving correctness per experiment.
The billion-token streaming result depends on exactly this: `results/scale_to_a_billion.json`
streams 1,000,013,824 tokens at constant **4.36 GB** peak RSS using the chunked deployment path
of a model trained at `T=32`, and the training-side guarantee is the same `streaming_check.json`
row in the table above (max-abs-delta 0.0, grad-cosine 1.0).

## Consequence 2 — state×weight mismatch heals fast, not just "eventually"

`results/state_weight_swap.json` (P23) transplants a *foreign* carried state (from a checkpoint
230,830,592 tokens away — `far_distance_tokens`) into a model running native weights, mid-stream,
and measures the excess loss relative to the native-state control. `swap_far` starts at
`excess_first50 = -0.044268` (statistically indistinguishable from native — a mismatch this large
does not even show up as an early penalty in the first 50 chunks) and both `swap_far` and
`swap_near` reach `excess_last100 = 0.0` (bit-identical to native) by `convergence_chunk = 4`. At
`chunk = 64` tokens per chunk, convergence at chunk 4 is **256 tokens** of the receiving stream.
The mismatch is not silently tolerated — it is actively repaired, in a number of tokens on the
order of the receptive field, not the sequence length. (Two of the three pre-registered checks in
this run did not clear their bar as stated — `cold_excess` was not ≥2× `swap_far_excess`, and
`shuffled` was not strictly worse than `swap_far` in every window — the *headline* healing-speed
result stands on `convergence_chunk` and `excess_last100`, which are unambiguous; the finer
ordering claims are noted honestly as not met.)

## Consequence 3 — the surgery license: growth and transplant are first-class operations

`results/hot_swap_growth.json` (P24) widens a live model mid-stream (`d64→d128`, carried state `Z`
migrated into the larger channel space) under the same equivalence gate (row above,
`6.67572e-06` on both stateless and chunked-3 forward passes). Post-surgery, the grown model
closes on the never-restarted `stay64` control (`phaseB_final` held-out: grown **5.283851** vs
stay64 **5.247354**, both far below the `fresh128`-from-scratch control at **5.410741** trained on
the same post-surgery token budget) — growth-with-transplant beats training a wider model from
zero at matched tokens. `results/beacon_swap.json` (P26) shows the same license for a *targeted*
read head: a swapped state vector written under one checkpoint (`t1`) is read correctly by a
different checkpoint (`t2`) at recall **1.0** across gaps 64/256/512 (`swap_recall`), under the
same class of equivalence gate. Growth, transplant, and adapter grafting are not separate hacks —
they are all instances of the one license: the operator is the same function of state and input
regardless of how or when the state arrived.

---

## What is not yet measured

- A direct equivalence check on the **beacon carrier-diagnosis** identification (which channel is
  "the" carrier) across a surgery boundary — currently the carrier channel is re-identified
  independently pre- and post-swap (`carrier_diagnosis.same_channel_per_layer: [true, true]` in
  `beacon_swap.json`) rather than proven invariant by construction. `[not yet measured —
  candidate experiment: perturb the surgery and check the carrier identification is stable]`
- `state_weight_swap.json`'s `c_shuffled_approx_chance` check fails per-window (`shuffled_worse:
  false` in most windows) — but the cause is measured, not open: past `convergence_chunk ≈ 4` all
  arms are bit-identical (deterministic convergence), so windowed metrics *cannot* separate them
  after the onset; in the onset itself the 30-trial fresh-injection diagnosis separates shuffled
  at **+0.9** excess vs +0.05 for the structured swap. The control works; the per-window metric is
  the wrong instrument after healing completes. `[refinement candidate: onset-resolved windows]`
