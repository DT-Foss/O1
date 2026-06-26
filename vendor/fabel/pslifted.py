"""
pslifted.py — two more tools from the Foss papers: O(1) relevance propagation
and a spectral graph-health metric.

TIER 3 — PS-Lifted relevance propagation (Constant-Round Gossip Consensus,
Foss 2026). To answer "what is related to X across the WHOLE graph", diffuse a
relevance field from X. A plain random walk takes O(1/gap) steps to cross a
bottleneck; the Fiedler-oriented lifted chain crosses in O(1) rounds regardless
of graph size — the same momentum-through-the-bottleneck mechanism that gives
O(1) gossip consensus. Same profile as fabel: deterministic, CPU-only, no GPU.

TIER 4 — Ginibre graph-health metric (One Constant / Universal Phase Transition,
Foss 2026). A directed Markov chain's complex spectrum has a phase transition:
an ordered (near-real, structured) regime vs. a disordered (Ginibre, plane-
filling, noisy) one. The mean nearest-neighbour eigenvalue spacing ⟨s2⟩ relative
to the Ginibre kernel value 1.0875 is a deterministic read on whether the causal
graph is clean structure or noise — a graph-quality signal with no ground truth
needed.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp


GINIBRE_KERNEL = 1.0874686665260935   # <s2> for plane-filling (disordered) spectra


def _index(edges: List[Dict]):
    nodes = sorted({e["a"] for e in edges} | {e["b"] for e in edges})
    return nodes, {n: i for i, n in enumerate(nodes)}


def relevance(edges: List[Dict], seed: str, pc: float = 0.65,
              rounds: int = 14, balance: bool = True) -> List[Tuple[str, float]]:
    """PS-Lifted relevance field from `seed`: who does this concept reach, and how
    strongly, across the whole graph. Fiedler-oriented momentum carries mass
    through bottlenecks in O(1) rounds (default 14, the paper's constant). Returns
    entities ranked by relevance, seed excluded.

    `balance` Sinkhorn-rebalances edge weights first (Foss Singularity Theorem):
    without it the diffusion collapses onto high-degree hubs (Jensen runaway), so
    every seed's "relevant" set degenerates to the same few hubs. Balancing keeps
    the field specific to the seed."""
    if balance:
        edges = sinkhorn_balance(edges)
    nodes, idx = _index(edges)
    n = len(nodes)
    if seed not in idx or n < 3:
        return []

    # symmetric adjacency + Fiedler orientation
    rows, cols, vals = [], [], []
    for e in edges:
        i, j = idx[e["a"]], idx[e["b"]]
        if i == j:
            continue
        w = float(e.get("conf", 1.0)) or 1.0
        rows += [i, j]
        cols += [j, i]
        vals += [w, w]
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    L = sp.diags(deg) - A
    try:
        import scipy.sparse.linalg as spla
        _, vecs = spla.eigsh(L, k=2, sigma=0, which="LM")
        fied = vecs[:, 1]
    except Exception:
        fied = np.zeros(n)

    # directed lifted transition: forward along Fiedler order with momentum pc
    Adir = sp.lil_matrix((n, n))
    A = A.tocoo()
    for i, j, w in zip(A.row, A.col, A.data):
        if fied[i] <= fied[j]:           # forward edge — momentum
            Adir[i, j] += w * pc
        else:                            # backward — reduced flow
            Adir[i, j] += w * (1.0 - pc)
    Adir = Adir.tocsr()
    rsum = np.asarray(Adir.sum(axis=1)).ravel()
    rsum[rsum == 0] = 1.0
    P = sp.diags(1.0 / rsum) @ Adir

    # diffuse a unit mass from the seed for `rounds` steps (O(1) in graph size)
    x = np.zeros(n)
    x[idx[seed]] = 1.0
    acc = np.zeros(n)
    for _ in range(rounds):
        x = P.T @ x
        acc += x
    acc[idx[seed]] = 0.0
    order = np.argsort(-acc)
    return [(nodes[i], float(acc[i])) for i in order if acc[i] > 1e-9]


def pf_centrality(edges: List[Dict], damping: float = 0.9,
                  iters: int = 100) -> Dict[str, float]:
    """Perron-Frobenius importance: the stationary distribution π of the graph's
    transition operator — the left eigenvector for eigenvalue 1 ("Born rule
    (structure): Perron-Frobenius replaces Born", Collapse Is Contraction, T01).
    π is the principled centrality of every concept: where probability mass
    accumulates under the graph's own dynamics, replacing naive degree counts.
    Damped (PageRank-style) so dangling/periodic structures still converge.
    Deterministic power iteration."""
    nodes, idx = _index(edges)
    n = len(nodes)
    if n == 0:
        return {}
    rows, cols, vals = [], [], []
    for e in edges:
        rows.append(idx[e["a"]])
        cols.append(idx[e["b"]])
        vals.append(float(e.get("conf", 1.0)) or 1.0)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    W = sp.diags(1.0 / deg) @ A
    pi = np.full(n, 1.0 / n)
    for _ in range(iters):
        nxt = damping * (W.T @ pi) + (1.0 - damping) / n
        nxt /= nxt.sum()
        if np.abs(nxt - pi).sum() < 1e-12:
            pi = nxt
            break
        pi = nxt
    return {nodes[i]: float(pi[i]) for i in range(n)}


def sinkhorn_balance(edges: List[Dict], iters: int = 30) -> List[Dict]:
    """Sinkhorn rebalancing of edge weights toward doubly stochastic — the
    collapse-prevention mechanism of the Foss Singularity Theorem (Emergent
    Gravity, Foss 2026): unbalanced propagation suffers a Jensen-convexity
    runaway onto high-degree hubs (measured sensitivity 1408× in the paper);
    periodic Sinkhorn projection is the ONLY mechanism preventing that collapse.
    Here: alternate row/column normalization of the confidence-weighted adjacency
    so no hub dominates the flow, then write the balanced weights back as each
    edge's `conf`. Deterministic."""
    nodes, idx = _index(edges)
    n = len(nodes)
    if n < 3:
        return edges
    M = np.zeros((n, n))
    for e in edges:
        M[idx[e["a"]], idx[e["b"]]] = max(
            M[idx[e["a"]], idx[e["b"]]], float(e.get("conf", 1.0)) or 1.0)
    mask = M > 0
    for _ in range(iters):
        rs = M.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        M = M / rs
        cs = M.sum(axis=0, keepdims=True)
        cs[cs == 0] = 1.0
        M = M / cs
        M[~mask] = 0.0          # keep the sparsity pattern — no invented edges
    out = []
    for e in edges:
        out.append(dict(e, conf=float(M[idx[e["a"]], idx[e["b"]]])))
    return out


def graph_health(edges: List[Dict]) -> dict:
    """Ginibre phase-transition read on the graph's directed spectrum. Returns
    `s2` (mean nearest-neighbour eigenvalue spacing), `ginibre_ratio` =
    s2/1.0875, and `order` ∈ [0,1] where 1 = perfectly ordered (real spectrum,
    clean causal structure) and 0 = fully disordered (Ginibre, noise)."""
    nodes, idx = _index(edges)
    n = len(nodes)
    if n < 8:
        return {"s2": 0.0, "ginibre_ratio": 0.0, "order": 1.0, "n": n}
    rows, cols, vals = [], [], []
    for e in edges:
        rows.append(idx[e["a"]])
        cols.append(idx[e["b"]])
        vals.append(float(e.get("conf", 1.0)) or 1.0)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    Pmat = sp.csr_matrix(sp.diags(1.0 / deg) @ A)
    ev = np.linalg.eigvals(Pmat.toarray())
    ev = ev[np.argsort(-np.abs(ev))][1:]      # drop the Perron λ1≈1
    if len(ev) < 3:
        return {"s2": 0.0, "ginibre_ratio": 0.0, "order": 1.0, "n": n}
    # nearest-neighbour spacing in the complex plane, normalized to mean 1
    nn = []
    for i in range(len(ev)):
        d = np.abs(ev - ev[i])
        d[i] = np.inf
        nn.append(d.min())
    nn = np.array(nn)
    mean = nn.mean() or 1.0
    s = nn / mean
    s2 = float((s ** 2).mean())
    ratio = s2 / GINIBRE_KERNEL

    # HONEST GUARD: the Ginibre ⟨s²⟩≈1.09 statistic is only meaningful for an
    # UNFOLDED spectrum in the random-matrix universality class. A sparse,
    # structured causal graph is NOT in that class — its eigenvalues cluster near
    # zero with a few outliers, so the raw spacing ⟨s²⟩ lands far outside the
    # valid [≈0.5, ≈2.0] band (Poisson↔ordered). When it does, the metric does not
    # apply; report that rather than a fabricated order score. (Proper unfolding +
    # a dense-enough graph would be needed to use this as intended.)
    applicable = 0.4 <= s2 <= 2.5
    order = (float(max(0.0, min(1.0, 1.5 - ratio))) if applicable else None)
    return {"s2": round(s2, 4), "ginibre_ratio": round(ratio, 3),
            "order": order, "applicable": applicable, "n": n}
