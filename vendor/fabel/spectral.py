"""
spectral.py — the Fiedler vector of the knowledge graph, applied to ranking.

Grounded in the Foss Gap Theorem (Foss, 2026): the Fiedler vector v2 (the
eigenvector of the second-smallest Laplacian eigenvalue) is a global ordering of
the graph that cuts it at its primary bottleneck — the sparsest split between two
otherwise-separate communities. In a causal knowledge graph mined from many
papers, those communities are REGIONS OF THE LITERATURE (obstetric vs.
cardiovascular, say).

This gives a principled novelty measure for a cross-paper hypothesis A→C: how far
apart are A and C in the Fiedler ordering? A hypothesis whose cause and effect sit
on OPPOSITE sides of the cut bridges two regions the literature keeps separate —
a genuine cross-domain lead. One whose endpoints sit in the same region is
within-community, i.e. obvious. The frequency-based 'surprise' it replaces was
degenerate (every bridge had near-equal frequency on a small corpus); the Fiedler
gap is structural and discriminative.

Deterministic, computed once (Lanczos, O(n) for sparse graphs). No model.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.sparse.csgraph as csg


def largest_component(edges: List[Dict]) -> List[Dict]:
    """The edges of the largest connected component. The Fiedler structure lives
    in the giant component; a disconnected graph has lambda2=0 (degenerate), so a
    long tail of isolated pieces would make the whole-graph Fiedler vector inert.
    Restricting to the giant component is the standard fix."""
    nodes = sorted({e["a"] for e in edges} | {e["b"] for e in edges})
    if len(nodes) < 3:
        return edges
    idx = {n: i for i, n in enumerate(nodes)}
    r, c = [], []
    for e in edges:
        if e["a"] != e["b"]:
            r += [idx[e["a"]], idx[e["b"]]]
            c += [idx[e["b"]], idx[e["a"]]]
    A = sp.csr_matrix(([1] * len(r), (r, c)), shape=(len(nodes), len(nodes)))
    _, lab = csg.connected_components(A, directed=False)
    if lab.max() == 0:
        return edges
    biggest = np.bincount(lab).argmax()
    keep = {nodes[i] for i in range(len(nodes)) if lab[i] == biggest}
    return [e for e in edges if e["a"] in keep and e["b"] in keep]


def fiedler(edges: List[Dict]) -> Tuple[Dict[str, float], float]:
    """Fiedler coordinate per entity, from the symmetrized causal graph, computed
    on the LARGEST CONNECTED COMPONENT (where the spectral structure lives).

    Each edge is {a, b, ...}. Returns (coord, lambda2): `coord[entity]` is the
    entity's position in the Fiedler ordering (normalized to [-1, 1]) for nodes in
    the giant component, and `lambda2` is its algebraic connectivity (>0 => a real
    bottleneck exists to bridge)."""
    edges = largest_component(edges)
    nodes = sorted({e["a"] for e in edges} | {e["b"] for e in edges})
    if len(nodes) < 3:
        return {n: 0.0 for n in nodes}, 0.0
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    # symmetric weighted adjacency (causal direction dropped for the cut — the
    # bottleneck is an undirected structural property)
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
    L = sp.diags(deg) - A

    # smallest two eigenpairs of the (symmetric PSD) Laplacian; v2 is the Fiedler
    # vector. shift-invert around 0 is robust for the near-singular low end.
    k = min(2, n - 1)
    try:
        vals_, vecs_ = spla.eigsh(L, k=k + 1, sigma=0, which="LM")
    except Exception:
        # dense fallback for tiny / pathological graphs
        vals_, vecs_ = np.linalg.eigh(L.toarray())
    order = np.argsort(vals_)
    lam2 = float(vals_[order[1]]) if len(order) > 1 else 0.0
    v2 = vecs_[:, order[1]] if vecs_.shape[1] > 1 else vecs_[:, order[0]]

    m = np.max(np.abs(v2)) or 1.0
    v2 = v2 / m                       # normalize to [-1, 1]
    coord = {nodes[i]: float(v2[i]) for i in range(n)}
    return coord, lam2


def gap_surprise(cause: str, effect: str, coord: Dict[str, float]) -> float:
    """Structural novelty of a hypothesis A→C: half the Fiedler distance between
    its endpoints, in [0, 1]. ~1 => endpoints on opposite sides of the primary
    cut (cross-community lead); ~0 => same region (obvious)."""
    ca, ce = coord.get(cause), coord.get(effect)
    if ca is None or ce is None:
        return 0.0
    return min(1.0, abs(ca - ce) / 2.0)


def communities(coord: Dict[str, float]) -> Dict[str, int]:
    """Two-way split at the Fiedler median — the primary community partition.
    Side 0 vs side 1; the cut between them is the graph's main bottleneck."""
    if not coord:
        return {}
    med = float(np.median(list(coord.values())))
    return {e: (0 if c <= med else 1) for e, c in coord.items()}
