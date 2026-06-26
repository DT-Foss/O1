# Why GSSM-Selective is length-invariant — the mechanism

**Companion to Contribution 4** (`README.md`). The empirical result: NoPE-Selective trained at
T=32 holds flat perplexity (×0.97, n=1; ×0.93 on an independent seed) out to T=8192 = 256×, while
the identical architecture *with* a positional encoding breaks (×4.23), and the gate-free "Pure"
variant explodes (×11–12). This note explains *why*, grounding each step in the actual recurrence.

The result is not a trained robustness — it is a **structural property of the operator**.

## The claim

The Selective scan contains **no absolute-position term**. Every index dependence enters only
through *relative lags* via the contraction. A causal recurrence with input-dependent gates and a
bounded state is therefore **shift-equivariant in the time index by construction**, so its readout
at step `t` is identical in distribution whether `t=20` or `t=8000`. A sinusoidal positional
encoding is the *only* term that references absolute `t`; beyond the training length it is
out-of-distribution, and it is the sole source of the break. The selective gate is what makes the
bounded state actually *use* the relative history; without it the state runs to its boundary.

Length-invariance requires **both** ingredients, and the data is a clean 2×2:

| | no PE | with PE |
|---|---|---|
| **selective gate** | **×0.97 (flat)** | ×4.23 (breaks) |
| **no gate (Pure)** | ×11.25 (explodes) | — |

## The argument, step by step

**1. Unroll: the state is a lag-weighted sum, with no `f(t)`.** The recurrence
`z_t = γ_t·z_{t−1} + α_t·φ(v̄_t)`, `φ(v)=log(1−v²)`, unrolls exactly to

```
z_t = Σ_{k=0..t}  α_k · Γ_{k→t} · φ(v̄_k),     Γ_{k→t} = ∏_{j=k+1..t} γ_j
```

Every factor — `α_k`, `φ(v̄_k)`, `γ_j` — is a function of **token content** `x_j`. The only
index-dependent factor is `Γ_{k→t}`, which depends on `t` and `k` **only through the intervening
tokens** — i.e. through the *lag*, never through the absolute coordinate `t`. There is no term
`g(t)` anywhere. **[PROVEN — read directly off the closed form.]**

**2. Shift-equivariance / Toeplitz by construction.** With no absolute-position term, the
per-path time–time kernel `K(s,t) = Σ_{k≤min(s,t)} α_k²·Γ_{k→s}·Γ_{k→t}` is supported on lags
only; in the constant-gate limit it collapses to the stationary Toeplitz kernel `γ^{|s−t|}`
(Ornstein–Uhlenbeck / Laplacian, positive-definite by Bochner). A model whose memory kernel is
lag-only has, by definition, the **same temporal receptive field at T=8192 as at T=32**.
**[PROVEN structurally.]** This is exactly Pillar P2 of the kernel-unification spine.

**3. The contraction makes the receptive field bounded *and* length-independent.** With
`γ_j ∈ (0,1)`, `Γ_{k→t}` decays geometrically in the lag. From the learned NoPE gates
(γ_mean≈0.225), the effective time constant is `τ = −1/log γ ≈ 1` token and the weight drops below
`1e−3` within **≈5–8 tokens** — far inside the T=32 training window. As `T` grows to 8192, no new
*kind* of computation appears: every readout still aggregates only its last handful of tokens,
exactly as in training. The contraction `τ<1` (finite DC gain `α/(1−γ) < ∞`, Pillar P1) guarantees
the receptive field is finite and **does not grow with `T`**. This is the mechanistic reason
"in-distribution at T=32" implies "in-distribution at T=8192." **[Structure PROVEN; the 5–8-token
horizon MEASURED from learned gates.]**

**4. The smoking gun: the gates do not move.** NoPE's learned gate statistics are **frozen across
256× extrapolation**: `γ_mean = 0.2252 → 0.2251` and `γ_p95 = 0.4352 → 0.4360` from T=32 to
T=8192 (four significant figures). The operator is *literally in-distribution* at every length
because it has no length-coupled quantity to drift. PPL even improves slightly (×0.97), the natural
gain from more left-context filling the short receptive field. **[MEASURED,
`results/length_extrap_v2_extreme.json` sat-stats.]**

**5. Why PE breaks it.** A sinusoidal PE adds `g(t)` to the embedding before the scan — the unique
injection of absolute position. Two failure modes, both visible in the data: (i) at `T>32` the
high-frequency components reach phases never seen in training (pure OOD input); (ii) the network
*adapts its gates* to exploit the in-training PE, and that adaptation is itself length-coupled —
the tell is +PE's `γ_mean` **drifting 0.231 → 0.356** as T grows, versus NoPE's flat 0.225.
Removing the PE removes the only length-dependent term and the only thing the gates compensate for.
**[Mechanism PROVEN; magnitude MEASURED.]**

**6. Why the gate is necessary (Pure).** Drop the selective gate and the state loses its bounded
contraction: effective `γ → 1`, DC gain `1/(1−γ) → ∞`, and `s_t = √(1−e^{z_t})` is driven to the
boundary `s*=1`. The data shows exactly this — Pure's last-position saturation climbs
**0.225 → 0.806** as T→8192, and PPL explodes ×11.25. With an unbounded receptive field the
"only-the-last-few-tokens" argument of Step 3 fails. So length-invariance needs **both**: no PE (no
absolute-position term) **and** the gate (bounded contraction that uses relative history).
**[MEASURED.]**

## Falsifier

The theory predicts NoPE-Selective's readout at step `t` depends only on the last `O(τ)≈5–8`
tokens, with **no `t`-coupled quantity**. It is falsified if either (a) any learned per-token
statistic drifts systematically with `T` under NoPE — it does **not** (γ frozen to 4 sig-figs
across 256×, Step 4); or (b) a hidden absolute-position term exists in the scan. *Code audit:* the
only `t`-indexed object in the scan layer's `forward` is the loop itself, which carries only the
state and per-token gates — no positional argument, no `t`-valued tensor enters the math. Position
lives exclusively in the parent LM's positional-encoding module, which NoPE replaces with identity.
The audit passes: there is no missed length-dependent term.

## What is proven vs measured

Steps 1–2 are **proven** (read off the closed form, code-audited). Step 3 is structurally proven
(contraction ⇒ bounded receptive field); the specific 5–8-token horizon is measured from the
learned gates. Steps 4–6 are measured. Step 5's *mechanism* is proven (a PE is the sole absolute-
position injector); its *magnitude* is measured. The one thing not derived from first principles is
the exact ×0.97 / ×4.23 / ×11.25 digits — those are empirical; the theory predicts their
**ordering and the near-elimination of drift under NoPE**, which is exactly what the 2×2 shows.
