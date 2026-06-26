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

---

## The causal knowledge engine — peer-reviewed foundations

O1's knowledge index (`vendor/fabel`, the `.causal` format) is not improvised. It is the
engineering core of a body of peer-reviewed work — nine papers accepted at four 2026 IEEE
conferences, plus the preprints behind them — all built on the same `.causal` deterministic,
zero-hallucination causal-inference substrate. The index O1 consults at runtime is the same
engine that has been validated across cryptanalysis, post-quantum security, nuclear knowledge
graphs, and autonomous discovery.

### Accepted at IEEE conferences (2026)

**IEEE-NANO 2026 — Nanjing, July 5–8 (the flagship conference of the IEEE Nanotechnology Council)**
- **Compression-Based Trust Verification of Lightweight Ciphers Deployed in Nano-IoT Communication
  Standards** — CASI distributional analysis applied to nano-scale IoT cipher trust.

**ICECET 2026 — Rome, July 6–9 (6th Intl. Conf. on Electrical, Computer and Energy Technologies)**
- **Deterministic Validation for Reliable LLM-Based Causal Knowledge Extraction** — the 14-step
  FOSS Gate: 88% precision on DocRED, 100% semantic F1 on causal samples, **100% byte-level
  determinism across 150 repeated extractions**, model-agnostic (Qwen-8B / Gemma-2B / Llama-3B all
  perfectly consistent despite 9× extraction-rate variation). This is the validation layer of the
  `.causal` pipeline. → preprint [10.5281/zenodo.18385710](https://doi.org/10.5281/zenodo.18385710)
- **Causal Graph Topology for Automated Security Margin Analysis and Blind Cipher Identification**
  → preprint [10.5281/zenodo.18591406](https://doi.org/10.5281/zenodo.18591406)
- **Compression Isolation of Distributional Signatures in NIST Post-Quantum Ciphertext** (ML-KEM /
  FIPS 203) → preprint [10.5281/zenodo.18601433](https://doi.org/10.5281/zenodo.18601433)
- **Persistent Cross-Round Carry Leakage in ARX Ciphers: Detection, Prediction, and Topological
  Classification** → preprint [10.5281/zenodo.18754499](https://doi.org/10.5281/zenodo.18754499)

**IEEE IRI 2026 — Seattle, July 31 – Aug 2 (27th Intl. Conf. on Information Reuse and Integration
for Data Science)**
- **The .causal Format: Deterministic Inference for AI-Assisted Hypothesis Amplification** — the
  binary knowledge-graph format with embedded inference that O1's index is built on.
  → preprint [10.5281/zenodo.18326222](https://doi.org/10.5281/zenodo.18326222)
- **Input-Agnostic Causal Knowledge Discovery**

**NURER 2026 — Almaty, Sept 10 (8th Intl. Conf. on Nuclear and Renewable Energy Resources, with IAEA)**
- **Backward Causal Inference on Nuclear Knowledge Graphs**
- **Spectral Signatures, PRNG, and Monte Carlo**

### Real-world assessment

- **Cryptographic Security Assessment of IBM z/OS Mainframe Infrastructure Using CASI Distributional
  Analysis** — a full security assessment of production IBM z/OS mainframe infrastructure (which
  processes ~87% of global credit-card transactions), run with a standard student account and no
  exploits. **50 findings.** RACF Legacy DES password hashing measured at **42.17 bits of effective
  entropy** (vs. the nominal 56) — crackable in **7.6 minutes on a consumer GPU for $0.08** — and
  the model was validated **bit-for-bit against a real IBM z15 running z/OS V2.5 (4/4 perfect
  match)**. Responsible disclosure to IBM PSIRT.
  → [10.5281/zenodo.18755826](https://doi.org/10.5281/zenodo.18755826) ·
  [SSRN 6298178](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6298178)

### The discovery engine

- **From Signal Amplification to Autonomous Discovery: The Sovereign Gap-Driven Knowledge Engine**
  — the gap-detection → query → retrieve → ingest loop that grows the graph autonomously; the
  symbolic ancestor of O1's runtime retrieval (Contribution 6).
  → [10.5281/zenodo.18336249](https://doi.org/10.5281/zenodo.18336249)
- **Sovereign Causal Graph: A Neuro-Symbolic Architecture for Air-Gapped Causal Knowledge Discovery**
  → [10.5281/zenodo.18287728](https://doi.org/10.5281/zenodo.18287728)

### Domain reach (`.causal` applied beyond ML)

The same deterministic engine has produced hypothesis-amplification work in biomedicine —
SIRT1-PGC-1α convergence, SSRIs as dual-mechanism therapy, Drp1 mitochondrial stability,
cross-species innate immunity, applied to Long COVID — evidence that `.causal` is a general
inference substrate, not a one-domain tool. (Zenodo: 18318310, 18311125, 18317350, 18326132,
and the format paper 18326222.)

### Software

- **dotcausal** — the canonical `.causal` binary knowledge-graph format with embedded inference,
  zero-hallucination by construction. [dotcausal.com](https://dotcausal.com) ·
  [github.com/dotcausal/dotcausal](https://github.com/dotcausal/dotcausal)
- **live-casi** — the CASI distributional-analysis tool used in the cryptanalysis work (the
  reproducibility backend for the cipher/PQC/mainframe results).
