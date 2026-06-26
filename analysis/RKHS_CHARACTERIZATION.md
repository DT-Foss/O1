# An RKHS / Reproducing-Operator Characterization of the GSSM Selective Scan

**Project:** GSSM — *From Markov Chains to Minkowski Space* (Foss 2026)
**Author:** functional-analysis / ML-theory lead
**Scope:** the precise, correct kernel statement for the Selective scan
`z_t = γ_t z_{t-1} + α_t log(1 − v_gated_t²)`, `s_t = √(1 − exp z_t)`.

> **One-line thesis.** The Selective scan is an exact, closed-form **reproducing-kernel
> readout in log-complement (rapidity) space**, but its kernel is a *data-dependent,
> time-inhomogeneous (causal Volterra) kernel*, **not** a fixed Mercer kernel. Every
> per-input trajectory lives in a genuine RKHS; the **conditioning on the input path
> changes the geometry from token to token**. The correct claim is therefore stronger
> *and* narrower than "replace BPTT with ridge regression on a fixed Mercer kernel" —
> which is **false** for this architecture, and we say exactly why.

---

## 0. Notation and the exact recurrence

From `moebius_scan_transformer_selective.py`, per channel (we suppress the head/channel
index `(h,d)` throughout; everything is elementwise in those indices):

- token velocity `v_t = tanh(W_v x_t) ∈ (−1, 1)`
- value gate `g_t = sigmoid(W_gate x_t) ∈ (0, 1)`, gated velocity `v̄_t := g_t v_t`
- forget gate `γ_t = sigmoid(W_γ x_t) ∈ (0, 1)`
- input gate `α_t = sigmoid(W_α x_t) ∈ (0, 1)`

Define the **scalar drive**

$$
a_t \;=\; \alpha_t \,\log\!\bigl(1 - \bar v_t^{\,2}\bigr), \qquad \bar v_t = g_t\,v_t \in (-1,1).
$$

The state and readout are

$$
z_t \;=\; \gamma_t\, z_{t-1} + a_t,\quad z_{-1}=0,
\qquad
s_t \;=\; \sqrt{1 - \exp(z_t)} \in [0,1].
$$

**Correctness note (matches the code, not the prose).** The drive uses the *gated*
velocity `v̄_t = g_t v_t` inside the log — `α_t log(1 − (g_t v_t)²)`, **not**
`α_t log(1 − v_t²)`. FINAL_REPORT flags this as a required code-vs-prose fix; we carry
the gate explicitly so the feature map below is the one the model actually computes.

---

## 1. The feature map: log-complement / rapidity space

Define the scalar **feature map**

$$
\boxed{\;\phi(v) \;=\; \log\!\bigl(1 - v^2\bigr), \qquad \phi:(-1,1)\;\longrightarrow\;(-\infty,\,0].\;}
$$

`φ` is the unique (up to the additive readout) coordinate in which the nonlinear
sqrt-coupled state becomes an **affine** functional of the inputs. Three facts pin down
its meaning:

1. **It linearizes the boundary.** `s = √(1−e^{z})` and `1 − s² = e^{z}`, so
   `z = log(1 − s²) = φ(s)`. The map `φ` is exactly the change of coordinates that turns
   the multiplicative boundary constraint `s ∈ [0,1]` into the *additive* half-line
   `z ∈ (−∞, 0]`. The state `z_t` and the per-token drive `a_t` live in the **same**
   space — the image of `φ`. The readout is `s_t = φ^{-1}(z_t)` with
   `φ^{-1}(z) = √(1 − e^{z})`.

2. **It is rapidity.** Write `v = tanh ξ` (so `ξ = artanh v` is the *rapidity* of a
   velocity-`v` boost). Then
   `1 − v² = sech² ξ`, hence `φ(v) = log sech² ξ = −2 log cosh ξ`. So the feature is
   `−2 log cosh(rapidity)` — the additive, *length-additive* coordinate of the Lorentz
   boost group `SO(1,1)`. This is the analytic backbone of the "Minkowski space" framing:
   composing boosts adds rapidities, and the scan composes drives **additively in `φ`**.

3. **It is concave, even, and unbounded-below.** `φ(0)=0` (zero drive for a
   stationary token), `φ` is even, strictly decreasing in `|v|`, and `φ(v) → −∞` as
   `|v| → 1`. A token moving near the light cone (`|v̄_t| → 1`) injects an unboundedly
   negative drive; the input gate `α_t ∈ (0,1)` scales how much of it enters. The clamp
   `v̄² ≤ 0.999` in the code is the numerical regularization of this `−∞` pole.

So the *additive drive* is `a_t = α_t · φ(v̄_t)`, an **input-gated feature** in
rapidity space. This is the object the scan sums.

---

## 2. The state as an inner product: the reproducing-kernel readout

### 2.1 Unrolling the linear recurrence

Solve the first-order linear recurrence with `z_{-1}=0`:

$$
\boxed{\;z_t \;=\; \sum_{k=0}^{t} \Gamma_{k\to t}\; a_k
\;=\; \sum_{k=0}^{t} \Bigl(\textstyle\prod_{j=k+1}^{t}\gamma_j\Bigr)\,
\alpha_k\,\phi(\bar v_k),
\qquad
\Gamma_{k\to t} := \prod_{j=k+1}^{t}\gamma_j \ \ (\text{empty product}=1).\;}
$$

*(Verified to 5.5e-17 in float64 against the sequential scan — 4.77e-7 in the default
float32, benign FP-summation noise, zero algorithmic difference; same identity the
`parallel_scan.py` affine operator computes to 8.88e-16 in float64 / 4.77e-7 in float32.)*

This is a **weighted sum of features** `{φ(v̄_k)}_{k≤t}` with positive, causal,
multiplicative weights `Γ_{k→t} α_k`. Define the **weight vector** and the **feature
vector** up to time `t`:

$$
w_t \;=\; \bigl(\,\Gamma_{0\to t}\,\alpha_0,\ \Gamma_{1\to t}\,\alpha_1,\ \dots,\ \Gamma_{t\to t}\,\alpha_t\,\bigr)
= \bigl(\alpha_k \!\!\prod_{j=k+1}^t \gamma_j\bigr)_{k=0}^{t},
\qquad
\Phi_t \;=\; \bigl(\phi(\bar v_0),\dots,\phi(\bar v_t)\bigr).
$$

Then the state is literally an inner product:

$$
\boxed{\,z_t \;=\; \langle w_t,\ \Phi_t\rangle_{\mathbb{R}^{t+1}},
\qquad
s_t \;=\; \sqrt{1 - \exp\langle w_t,\Phi_t\rangle}\,.}
$$

The readout is the closed-form decoder `φ^{-1}` applied to a linear functional of the
features. No BPTT is required to *express* the map — only to *learn* the projections
`W_v, W_gate, W_γ, W_α` that generate `(w_t, Φ_t)`.

### 2.2 The reproducing kernel between two query times

The natural RKHS object is the kernel between two *times* `s, t` of the **same input
path** `x`. With the convention that `z_t = ⟨w_t, Φ_t⟩` and treating the drive features
as the coordinates of an element of a Hilbert space, the Gram between the two state
functionals is

$$
\boxed{\;
K^{(x)}(s,t)
\;=\; \sum_{k=0}^{\min(s,t)}
\underbrace{\Bigl(\alpha_k\!\!\prod_{j=k+1}^{s}\gamma_j\Bigr)}_{w_s[k]}
\underbrace{\Bigl(\alpha_k\!\!\prod_{j=k+1}^{t}\gamma_j\Bigr)}_{w_t[k]}
\;=\; \sum_{k=0}^{\min(s,t)} \alpha_k^2\;
\Gamma_{k\to s}\,\Gamma_{k\to t}.
\;}
$$

This is a **causal (lower-triangular) Volterra kernel** indexed by time, parameterized by
the input path `x` (through `γ_j(x), α_k(x)`). Its diagonal `K^{(x)}(t,t) = Σ_{k≤t}
α_k² Γ_{k→t}²` is the "leaky memory norm" of the state at time `t`.

> **Mercer reading (and its limit).** `K^{(x)}(s,t)` is, *for a fixed path* `x`, a Mercer
> kernel on the index set `{0,…,T}`: it is the Gram of the explicit feature vectors
> `{w_s}`, hence symmetric PSD (Section 3). The catch — and the whole point — is the
> superscript `(x)`: the kernel is **a different Mercer kernel for every input**, because
> `γ` and `α` are functions of `x`. There is **no single kernel `K(s,t)` on raw
> velocity space `v`** that reproduces the scan across inputs. See Section 3.3.

### 2.3 The constant-gamma collapse to a classical Mercer / Toeplitz kernel

When `γ_t ≡ γ` and `α_t ≡ α` are *constant in time* (input-independent), the weights
become a pure geometric kernel and the time-time kernel is **stationary**:

$$
K(s,t) \;=\; \alpha^2 \sum_{k=0}^{\min(s,t)} \gamma^{\,s-k}\gamma^{\,t-k}
\;=\; \alpha^2\,\gamma^{\,|s-t|}\,\frac{1-\gamma^{2(\min(s,t)+1)}}{1-\gamma^{2}}.
$$

For `min(s,t)` large this is `≈ \dfrac{\alpha^2}{1-\gamma^2}\,\gamma^{|s-t|}` — the
**exponential (Laplacian / Ornstein–Uhlenbeck) kernel** `γ^{|s−t|} = e^{-|s-t|/\tau}`
with time constant `τ = −1/\log γ`. This is the textbook stationary Mercer kernel of a
leaky integrator. In this regime the read map `z = K a` is a fixed lower-triangular
**Toeplitz geometric convolution** — exactly `constant_gamma_closed_form` in
`parallel_scan.py`. *This* is the only regime where "fixed Mercer kernel" is literally
true; the Selective architecture deliberately leaves it.

---

## 3. Positive semi-definiteness: the precise statement

There are three distinct Gram matrices one can write, and they have **three different**
PSD statuses. Conflating them is exactly the error of the overclaiming version.

### 3.1 Per-path time-time Gram `K^{(x)}(s,t)` — PSD, always

For a **fixed input path** `x`, the matrix `G_{st} = K^{(x)}(s,t)` is the Gram matrix of
the explicit real weight vectors `{w_t}_{t=0}^{T}` embedded in `ℝ^{T+1}` (pad `w_t` with
zeros for `k > t`). Any Gram of real vectors is symmetric PSD:

$$
\forall c\in\mathbb{R}^{T+1}:\quad
\sum_{s,t} c_s c_t\,K^{(x)}(s,t)
= \Bigl\|\textstyle\sum_t c_t\, w_t\Bigr\|_2^2 \;\ge\; 0.
$$

*Numerically confirmed:* min eigenvalue `0.42 > 0` for a random data-dependent `γ` path,
`0.32 > 0` for the constant-`γ` stationary kernel. **PSD holds per path with no
conditions** — it is the trivial Gram fact, and it is the correct foundation. The
*content* is not "is it PSD" (it must be) but **which kernel it is**, namely the
causal Volterra kernel of Section 2.2.

### 3.2 Constant-gamma stationary kernel — PSD by Bochner

In the collapse of Section 2.3, `K(s,t) = c·γ^{|s−t|}` with `γ ∈ (0,1)`, `c>0`. This is a
positive-definite function on `ℤ` by Bochner's theorem: its discrete-time Fourier
transform is the (strictly positive) Poisson kernel
`(1−γ²)/(1 − 2γ\cosω + γ²) ≥ 0`. So the constant-gamma case is **strictly Mercer-PD**,
not merely PSD. This is the regime where a classical "kernel ridge regression" reading is
fully rigorous.

### 3.3 Cross-input Gram — PSD as numbers, but NOT a Mercer kernel in `v`

Now the load-bearing distinction. Take `N` different inputs `x^{(1)},…,x^{(N)}` and the
final states `z_T^{(i)} = ⟨w_T^{(i)}, Φ_T^{(i)}⟩`. The cross-sample Gram
`G_{ij} = ⟨ψ_i, ψ_j⟩` of any chosen real feature vectors `ψ_i` is **trivially PSD**
(min eigenvalue `−9.7e-17 ≈ 0`, machine precision — confirmed). **But this PSD-ness buys
nothing**, because:

> The weight vector `w_T^{(i)} = (α_k^{(i)} Π γ_j^{(i)})_k` and the feature vector
> `Φ_T^{(i)} = (φ(v̄_k^{(i)}))_k` **both depend on `x^{(i)}`**. There is no fixed map
> `Ψ: (\text{raw velocities } v) ↦ \mathcal H` and no fixed kernel `k(v, v')` such that
> `z_T^{(i)} = ⟨Ψ(v^{(i)}), \beta⟩` reproduces the scan across inputs with a single
> weight `β`. The "kernel trick" requires the feature map to be **input-only**; here the
> *weighting* (`γ, α`) is input-dependent, so the geometry itself is reindexed per token.

Formally: the scan is a **bilinear-in-its-own-gates** operator. Writing it as
`z_T = Σ_k W(x)_k · φ(v̄_k(x))` makes the "kernel weights" `W(x)_k = α_k(x) Π_{j>k}
γ_j(x)` *functions of the same `x`* that produces the features. That is precisely a
**time-inhomogeneous / data-controlled kernel**, the discrete analogue of a Volterra
series with input-dependent memory, **not** a stationary Mercer kernel. The map
`x ↦ z_T` is therefore in the class of *gated/controlled kernels* (cf. signature kernels,
input-dependent RKHS), where the reproducing property holds **only after conditioning on
the gate path**.

**Conclusion (PSD).**
- **PSD per path:** yes, unconditionally (Gram of real weight vectors). *This is the
  correct and publishable kernel statement.*
- **Mercer / Bochner-PD:** yes, **iff** `γ` (and `α`) are constant in time — the
  exponential-kernel leaky integrator. The architecture's selectivity is exactly the
  departure from this.
- **Fixed Mercer kernel in raw velocity across inputs:** **no.** This is where the prior
  chat overclaimed. The kernel is time-inhomogeneous because the gates depend on `x`.

---

## 4. The affine operator is the kernel's composition law

`parallel_scan.py` proves the recurrence is exactly associative under

$$
(A_2,B_2)\otimes(A_1,B_1) = (A_2 A_1,\ A_2 B_1 + B_2),\quad \text{identity } (1,0),
$$

with `(A_t,B_t) = (γ_t, a_t)`. In kernel language: `A_t = γ_t` is the **propagator** of
the memory weight (it transports `w` forward one step, `Γ_{k→t} = Γ_{k→t-1}·γ_t`), and
`B_t = a_t` is the **new feature injected** at time `t`. The B-component of the inclusive
prefix is `z_t = ⟨w_t, Φ_t⟩` (Section 2.1). So:

- the **monoid** `(ℝ_{>0} × ℝ, ⊗)` is the algebra of the kernel;
- the **`A`-product** `Γ_{k→t}` is the kernel's time-inhomogeneous transition weight;
- the **`B`-accumulation** is the RKHS sum.

Associativity ⇒ the inner product `⟨w_t, Φ_t⟩` is computable in `O(\log T)` *depth*
(Hillis–Steele / Blelloch). This is the algorithmic content: the reproducing-kernel
readout is a **prefix scan over a semiring**, parallelizable, with the same per-path
Mercer structure at every prefix. *(Implementation status per FINAL_REPORT: the parallel
scan is verified to 8.88e-16 in float64 (4.77e-7 in float32, FP-summation noise) but the
deployed model still uses the `O(T)` sequential loop; the kernel statement is independent
of which is used.)*

---

## 5. Specializations: Mamba, S5, LRU are the same affine-operator readout

Every modern linear/selective SSM is a first-order recurrence
`h_t = A_t h_{t-1} + B_t u_t`, i.e. carries `(A_t, B_t)` under the **same** affine
operator `⊗`. They differ only in three parametric choices: **(i)** the algebra of `A`
(real scalar vs real diagonal vs complex diagonal), **(ii)** whether `A_t` depends on the
input, and **(iii)** the input map producing `B_t`. The affine-operator / reproducing-
kernel view of Section 2–4 is the common generalization. *(The complex-diagonal case is
verified to reproduce the sequential recurrence exactly, error `0.0`.)*

| Model | State alg. | `A_t` (forget) | `B_t` (drive) | `A_t` input-dep? | Kernel type |
|---|---|---|---|---|---|
| **GSSM-Selective** | real **scalar**, `∈(0,1)` | `γ_t = σ(W_γ x_t)` | `α_t φ(v̄_t)`, `φ=log(1−·²)` | **yes** | data-dep. time-inhomog. Volterra, **nonlinear feature `φ`** |
| **Mamba (S6)** | real diagonal `∈(0,1)` | `\bar A_t=\exp(Δ_t A)` | `Δ_t \bar B\, u_t` | **yes** | data-dep. time-inhomog. (linear feature `u`) |
| **S5** | **complex** diagonal, `|·|<1` | `\exp(Δ Λ)` (time-const) | `Δ B\, u_t` | no | **fixed Mercer** (LTI), complex exp. kernel |
| **LRU** | **complex** diagonal `e^{-ν+iθ}` | `\diag(λ)` (time-const) | `B\, u_t` | no | **fixed Mercer** (LTI), complex exp. kernel |

Readings:

- **GSSM is to Mamba as `φ` is to identity.** Mamba's drive is the *linear* feature
  `B_t u_t`; GSSM's drive is the *nonlinear log-complement feature* `α_t φ(v̄_t)` with the
  closed-form sqrt readout `φ^{-1}`. Both are input-gated (selective), so both are
  **time-inhomogeneous** kernels — Mamba is **not** a fixed Mercer kernel either, for the
  same reason GSSM isn't. GSSM additionally guarantees `s_t ∈ [0,1]` *by construction*
  via `φ`/`φ^{-1}`; Mamba has no such boundedness.
- **S5 / LRU are the LTI Mercer case.** Their `A` is time-constant and input-independent,
  so they collapse to Section 3.2: a genuine fixed (complex-exponential) Mercer kernel,
  the linear-system convolution kernel. GSSM's constant-`γ` limit (Section 2.3) is the
  **real, scalar** sibling of the S5/LRU kernel — `γ^{|s−t|}`, a 1-pole real filter.
- **The unifying object** is the affine prefix scan over `(A,B)`; the "RKHS" is the image
  of the `B`-accumulation; "selective" ⇔ "`A_t` is input-dependent" ⇔ "the kernel is
  time-inhomogeneous, not Mercer."

---

## 6. The lead claim, and the precise boundary of the RKHS view

### 6.1 The one falsifiable claim to lead with

> **The Selective scan is an exact closed-form reproducing-kernel readout in
> rapidity space:** with feature map `φ(v)=log(1−v²)` and input-gated, forget-weighted
> coefficients `w_t[k] = α_k Π_{j>k} γ_j`, the state is the inner product
> `z_t = ⟨w_t, φ(v̄_{0:t})⟩` and the output is `s_t = √(1−exp z_t)`. The induced
> per-path time-time Gram `K^{(x)}(s,t) = Σ_{k≤\min(s,t)} α_k² Γ_{k→s}Γ_{k→t}` is **PSD
> for every input**, and it reduces to the **exponential Mercer kernel `γ^{|s−t|}`**
> exactly when the forget gate is input-independent — the regime that contains S5/LRU as
> the complex-diagonal LTI case.

**Falsifier (concrete):** freeze `W_γ, W_α` so `γ, α` are constant; then the trained
model's read map *must* coincide, to numerical tolerance, with kernel ridge / a
geometric Toeplitz convolution `γ^{|s−t|}` (Section 2.3), and BPTT through the scan must
be replaceable by a closed-form solve. If a constant-gate Selective model trained with
BPTT does **not** match the closed-form geometric-kernel readout, the characterization is
wrong. *(Predicted: it matches — `constant_gamma_closed_form` already reproduces the
sequential scan to 5e-17 in float64 (4.77e-7 in float32, FP-summation noise); the open
claim is that the **learned** constant-gate optimum is the kernel solution.)*

### 6.2 Where the precise claim differs from a fixed-Mercer reading

1. **Not a fixed Mercer kernel.** The load-bearing boundary: because `γ, α, gate` all
   depend on `x`, there is **no single kernel `k(v,v')`
   in raw velocity space across inputs.** "Replace BPTT with ridge regression on a fixed
   Mercer kernel" is **false** for the selective model. Ridge/closed-form solves apply
   only in the **constant-gate (LTI) special case** — which is exactly the case the
   architecture is built to leave.

2. **The reproducing property is conditional.** The clean RKHS picture holds **per gate
   path**, i.e. *after* the input has determined `{γ_j, α_k}`. You cannot precompute the
   Gram before seeing `x`; the kernel is *controlled by the same signal it acts on*
   (a Volterra / signature-kernel situation, not a kernel-machine situation).

3. **The readout is outside the RKHS norm.** `s_t = φ^{-1}(z_t) = √(1−e^{z_t})` is a
   **nonlinear decode** of the linear functional. Linearity (and hence the literal RKHS
   inner-product structure) lives in `z`-space, not in the observed `s`-space. Any kernel
   regression statement is about `z`, with `s` recovered by the fixed bijection `φ^{-1}`.

4. **`φ` is not a positive-definite-kernel feature on its own.** `φ(v)=log(1−v²)` is a
   1-D nonlinear coordinate; the PSD-ness comes from the **outer-product weighting**
   `w_t w_t^⊤`, not from a Moore–Aronszajn kernel on velocities. We do **not** claim
   `k(v,v') = φ(v)φ(v')` is a universal kernel; it is a rank-1 feature, and the memory
   structure is what makes the time-time Gram rich.

5. **Input-dependence breaks stationarity / shift-invariance.** Bochner's theorem
   (Section 3.2) only certifies PD in the constant-gate limit. For selective gates the
   kernel is non-stationary and the spectral (Fourier) characterization does not apply;
   PSD survives (Gram of real vectors) but the *spectral* PD guarantee does not.

6. **No exact associative recall from the scalar channel.** Consistent with
   FINAL_REPORT's 14% MQAR limit: a scalar (`d=1` per channel) reproducing readout cannot
   implement KV-binding. The RKHS view explains *why* — a rank-controlled scalar
   functional has bounded representational capacity for pair lookups — but it does **not**
   rescue recall. This is a capacity limit: a rank-controlled scalar functional has bounded
   pair-lookup capacity. One attention layer (rank K) fixes it; the standalone scalar limit
   is structural.

### 6.3 Why the correct claim is *stronger* than the hype

The overclaim ("fixed Mercer kernel ⇒ ridge regression replaces BPTT") is a **false
generality**. The true statement is a **sharper, verifiable structural result**: the
scan is the affine prefix scan of an input-controlled reproducing kernel, exactly
PSD per path, with `φ = log(1−v²)` as the rapidity feature and `φ^{-1}` as the
boundedness-guaranteeing decoder, specializing to S5/LRU/Mamba by three explicit
parametric switches. It tells you **precisely** when the kernel-machine shortcut is legal
(constant gates) and why selectivity forbids it (input-dependent memory ⇒
time-inhomogeneity) — which is more useful, and true.

---

## Appendix A — derivation of the constant-gate attractor in kernel form

With constant gates, `z_t = α φ(v̄) Σ_{k=0}^{t} γ^{t-k} → α φ(v̄)/(1−γ)` as `t→∞`
(geometric series, `γ<1`). Hence the interior attractor

$$
s^\* = \sqrt{1 - \exp\!\Bigl(\tfrac{\alpha}{1-\gamma}\,\phi(v̄)\Bigr)}
= \sqrt{1 - (1-\bar v^2)^{\alpha/(1-\gamma)}} \in (0,1),
$$

matching FINAL_REPORT exactly (numeric vs closed form: error `0.0`). The exponent
`α/(1−γ)` is the **DC gain of the kernel** `Σ_k γ^k = 1/(1−γ)` times the input gate — the
stationary value of the leaky-integrator Mercer kernel applied to a constant feature.
The interior (`s^\*<1` whenever `|v̄|<1`) vs boundary (`s^\*=1`) attractor dichotomy of the
project is thus a statement about the **DC gain of the reproducing kernel**: finite gain
`α/(1−γ) < ∞` ⇒ interior; the Pure variant's unbounded effective gain ⇒ boundary
saturation.

## Appendix B — the kernel as a controlled (Volterra) operator, formally

Let `K^{(x)}: ℓ^∞_{[0,T]} → ℝ^{T+1}` send the feature sequence `Φ = (φ(v̄_k))_k` to the
state sequence `z = (z_t)_t` via the lower-triangular operator
`[K^{(x)}]_{t,k} = α_k Γ_{k→t}\,\mathbf 1[k≤t]`. Then `z = K^{(x)} Φ`, and the time-time
Gram of Section 2.2 is `K^{(x)} (K^{(x)})^⊤` restricted appropriately — manifestly PSD.
The dependence `x ↦ K^{(x)}` (through `γ_j(x), α_k(x)`) is what places this in the class
of **controlled / data-dependent kernels** rather than fixed Mercer kernels. The map
`x ↦ z` is a **causal, gain-bounded, nonlinear (through `φ` and the gates) Volterra
operator** whose 1st-order (constant-gate) truncation is exactly the LTI exponential
kernel of S5/LRU.

---

### Verification log (all numerical claims in this document)

- Unrolled inner-product form `z_t = Σ_k Γ_{k→t} a_k` vs sequential scan: **max err 5.5e-17 (float64); 4.77e-7 (float32, FP-summation noise, zero algorithmic difference)**.
- Constant-gate attractor `√(1−(1−v²)^{α/(1−γ)})` vs 5000-step iterate: **err 0.0**.
- Constant-`γ` time-time Gram min eigenvalue: **+0.32** (Mercer-PD).
- Data-dependent-`γ` per-path time-time Gram min eigenvalue: **+0.42** (PSD per path).
- Cross-sample state Gram min eigenvalue: **−9.7e-17 ≈ 0** (trivially PSD as real Gram;
  not a fixed kernel in `v`).
- Complex-diagonal (S5/LRU) affine operator vs sequential recurrence: **err 0.0**.
