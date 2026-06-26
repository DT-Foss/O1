"""
contraction.py — two confidence operators from the Foss papers, each applied
where it is actually correct (and not where it isn't).

The two operations confidence undergoes in a causal graph pull in OPPOSITE
directions, and the papers give the right tool for each:

  CHAINING A→B→C  — the inference must get LESS certain. This is a CONTRACTION.
    "Collapse Is Contraction" (Foss, 2026) makes the Birkhoff/Dobrushin
    contraction coefficient τ<1 the order parameter of the classical phase. We
    use the graph's MEASURED Dobrushin coefficient as the chain-decay rate, in
    place of the ad-hoc 0.85 — the decay is now a property of the data, not a
    guess.

  EVIDENCE FUSION (k papers each assert A→B) — the claim must get MORE certain,
    approaching but never reaching 1. This is exactly the Möbius/Lorentz velocity
    addition f(λ,v)=(λ+v)/(1+λv) (Markov→Minkowski, Foss 2026): two strong
    independent confirmations combine to very-high-but-<1, never overshooting.

The common error is to reach for the elegant Möbius identity at the chaining
step. That is BACKWARDS — velocity addition moves toward the limit, chaining must
move away from it. Möbius belongs to fusion, contraction to chaining.
"""
from __future__ import annotations

from typing import Dict, List


def dobrushin(edges: List[Dict]) -> float:
    """The graph's per-hop confidence contraction = the SPECTRAL contraction rate
    (magnitude of the second eigenvalue of the row-stochastic transition matrix).

    NOTE on why not the literal Dobrushin coefficient: τ_Dobrushin =
    (1/2)max||P_i−P_j||_1 is DEGENERATE on a sparse causal graph — any two nodes
    with disjoint successors give ||·||_1 = 2, so τ = 1 (no contraction) almost
    always. Measured 1.000 on the medical graph: useless as a decay. The spectral
    second eigenvalue |λ2| is the actual asymptotic mixing rate (the
    slowest-decaying mode, the Birkhoff/Hilbert-metric contraction "Collapse Is
    Contraction" points to) and stays < 1 on a connected aperiodic component.
    Returns |λ2| ∈ (0,1): the fraction of confidence a chain retains per hop."""
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    nodes = sorted({e["a"] for e in edges} | {e["b"] for e in edges})
    if len(nodes) < 3:
        return 0.85
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    rows, cols, vals = [], [], []
    for e in edges:
        rows.append(idx[e["a"]])
        cols.append(idx[e["b"]])
        vals.append(float(e.get("conf", 1.0)) or 1.0)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    P = sp.diags(1.0 / deg) @ A          # row-stochastic transition matrix
    try:
        # two largest-magnitude eigenvalues; λ1≈1, λ2 is the contraction rate
        ev = spla.eigs(P, k=min(2, n - 2), which="LM",
                       return_eigenvectors=False, maxiter=2000)
        mags = sorted(np.abs(ev), reverse=True)
        lam2 = float(mags[1]) if len(mags) > 1 else float(mags[0])
    except Exception:
        return 0.85
    return min(0.999, max(0.05, lam2))


def mobius_fuse(c1: float, c2: float) -> float:
    """Möbius/Lorentz addition of two independent confidences for the SAME claim:
    f(c1,c2) = (c1+c2)/(1+c1·c2). Stays in [0,1), approaches 1 with more
    agreement, never overshoots. Two 0.9 confirmations → 0.994; a 0.9 and a 0.3 →
    0.945. Use ONLY for fusing evidence of one edge, never for chaining."""
    c1 = max(0.0, min(0.999, float(c1)))
    c2 = max(0.0, min(0.999, float(c2)))
    denom = 1.0 + c1 * c2
    return (c1 + c2) / denom if denom else c1


def fuse_all(confidences: List[float]) -> float:
    """Fuse a list of independent confidences for one claim via Möbius addition
    (order-independent up to floating point)."""
    if not confidences:
        return 0.0
    acc = confidences[0]
    for c in confidences[1:]:
        acc = mobius_fuse(acc, c)
    return acc


def rapidity_fuse(confidences: List[float]) -> float:
    """Exact N-way evidence fusion in rapidity coordinates (Markov→Minkowski,
    Foss 2026): ψ = arctanh(c) is ADDITIVE, so fusing N independent confirmations
    is one sum — c_fused = tanh(Σ arctanh(c_i)).

    Mathematically this equals iterated pairwise Möbius addition (tanh addition
    theorem), so `fuse_all` already computes the same value; the rapidity form is
    the closed form: one pass, no iteration order at all, numerically stable via
    log1p. This is the principled "more papers agree → confidence → 1, never
    past it" operator."""
    import math
    total = 0.0
    for c in confidences:
        c = max(0.0, min(0.999999, float(c)))
        # arctanh(c) = 0.5*log((1+c)/(1-c)), via log1p for stability
        total += 0.5 * (math.log1p(c) - math.log1p(-c))
    return math.tanh(total)


def birkhoff_tau(edges: List[Dict], n_pairs: int = 100,
                 seed: int = 0) -> float:
    """The Birkhoff contraction coefficient τ of the graph's transition operator,
    estimated as in "Collapse Is Contraction" (Foss 2026, Eq. 3):

        τ(W) = sup_{x≠y}  dTV(Wx, Wy) / dTV(x, y)

    over random distribution pairs (the paper uses 100). τ is THE inference-
    safety order parameter: τ < 1 means multi-hop inference CONTRACTS (chains
    decay safely — the classical phase); τ → 1 means it preserves/amplifies
    differences (runaway amplification of noise — the PCR-contamination regime).
    Deterministic given the seed."""
    import numpy as np
    import scipy.sparse as sp

    nodes = sorted({e["a"] for e in edges} | {e["b"] for e in edges})
    n = len(nodes)
    if n < 3:
        return 0.0
    idx = {n_: i for i, n_ in enumerate(nodes)}
    rows, cols, vals = [], [], []
    for e in edges:
        rows.append(idx[e["a"]])
        cols.append(idx[e["b"]])
        vals.append(float(e.get("conf", 1.0)) or 1.0)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    deg = np.asarray(A.sum(axis=1)).ravel()
    # dangling nodes: self-loop so W stays stochastic
    dang = deg == 0
    deg[dang] = 1.0
    W = sp.diags(1.0 / deg) @ A
    if dang.any():
        W = W + sp.diags(dang.astype(float))

    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_pairs):
        x = rng.dirichlet(np.ones(n))
        y = rng.dirichlet(np.ones(n))
        d0 = 0.5 * np.abs(x - y).sum()
        if d0 < 1e-12:
            continue
        d1 = 0.5 * np.abs(W.T @ x - W.T @ y).sum()
        worst = max(worst, d1 / d0)
    return float(min(1.0, worst))


def safe_depth(tau: float, signal_floor: float = 0.05) -> int:
    """How many inference hops stay meaningful on a graph with contraction τ: the
    depth k where τ^k first drops below `signal_floor`. τ→1 ⇒ unbounded depth is
    claimed safe by decay alone — which is exactly when amplification risk is
    highest, so we CAP at 8 and flag. τ small ⇒ chains die fast; don't infer
    deeper than the signal survives."""
    import math
    if tau >= 0.999:
        return 8     # decay won't protect you here — hard cap
    if tau <= 0.0:
        return 1
    return max(1, min(8, int(math.log(signal_floor) / math.log(tau))))
