# Foundational Papers

GSSM is the engineering instantiation of a body of mathematical work by David Tom Foss on
Möbius coupling, doubly-stochastic spectra, and non-reversible lifted Markov chains. Every
component of this repository traces to a result in these papers — the geometry, the bounded
state, the parallel scan, the spectral structure that informs the recall work. All are
publicly archived with permanent DOIs.

## The architecture

- **Geometric State Space Models: Bounded Hyperbolic Recurrence for Language Modeling** — David Tom Foss.
  ResearchGate: https://www.researchgate.net/publication/407219530
  *(The GSSM architecture paper — the direct basis of this repository.)*

## The mathematical foundation

| Paper | DOI / link |
|---|---|
| **From Markov Chains to Minkowski Space**: Lorentz Invariance, Quantum Measurement, and Gravitational Analogs | [10.5281/zenodo.18686982](https://doi.org/10.5281/zenodo.18686982) |
| **One Constant Rules All 2D Spectra**: Universal Convergence to the Ginibre Kernel | [10.5281/zenodo.19055912](https://doi.org/10.5281/zenodo.19055912) |
| **The Foss Number**: F = 1 + 1/(3π) as the Asymptotic Second Moment of Eigenvalue Spacings in Random Doubly Stochastic Matrices | [10.5281/zenodo.19024376](https://doi.org/10.5281/zenodo.19024376) |
| **Collapse Is Contraction**: The Foss Interpretation of Quantum Mechanics | [10.5281/zenodo.18944821](https://doi.org/10.5281/zenodo.18944821) |
| **Unitarity Is the Boundary**: A Complete Classification of Quantum Phenomena Reproducible by Classical Non-Reversible Dynamics | [10.5281/zenodo.18943317](https://doi.org/10.5281/zenodo.18943317) |
| **Non-Reversibility Is All You Need**: Classical Super-Quantum Speedup via Lifted Markov Chains | [10.5281/zenodo.18939761](https://doi.org/10.5281/zenodo.18939761) |
| **Emergent Gravity, Biological Intelligence, and Standard Model Predictions** from Möbius-Coupled Doubly Stochastic Lattices | [10.5281/zenodo.18939932](https://doi.org/10.5281/zenodo.18939932) |
| **Constant-Round Gossip Consensus**: Push-Sum on Non-Reversible Lifted Markov Chains for Decentralized Optimization | [10.5281/zenodo.18875923](https://doi.org/10.5281/zenodo.18875923) |
| **The Foss Gap Theorem**: Linear Cheeger Improvement via Fiedler-Oriented Lifted Markov Chains | [10.5281/zenodo.18945138](https://doi.org/10.5281/zenodo.18945138) |

Additional preprints on SSRN:
[abstract 6349438](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6349438) ·
[abstract 6265238](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6265238)

## How they connect to the code

- **From Markov Chains to Minkowski Space** — the Möbius–Lorentz correspondence on doubly-stochastic
  spectra; the geometry GSSM's bounded state lives in (`reference/`).
- **One Constant / The Foss Number** — the Ginibre kernel value ⟨s²⟩ = 1.0874 and the β=3 cubic
  repulsion of 2D spectra; the spectral structure behind the key-geometry experiments
  (`src/holographic_ginibre.py`, `analysis/RESEARCH_LOG.md`).
- **Constant-Round Gossip / Foss Gap Theorem** — PS-Lifted dynamics and the associative-scan lift;
  the basis of the `O(log T)` parallel scan (`src/parallel_scan.py`).
- **Collapse Is Contraction / Unitarity Is the Boundary** — the contraction τ<1 / bounded-state
  core that underwrites GSSM's stability guarantee.
