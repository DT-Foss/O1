# dotcausal

**The Knowledge Graph Format for AI**

[![PyPI version](https://badge.fury.io/py/dotcausal.svg)](https://badge.fury.io/py/dotcausal)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The `.causal` format is a binary knowledge graph format with **embedded deterministic inference**. It solves the fundamental problem of AI-assisted discovery: **LLMs hallucinate, databases don't reason**.

## Why .causal?

### The Problem

| Technology | What it does | What's missing |
|------------|--------------|----------------|
| **SQLite** | Stores facts | No reasoning - only returns explicit matches |
| **Vector RAG** | Finds similar text | No logic - returns relevance, not causality |
| **LLMs** | Reasons creatively | Hallucination risk - invents plausible but false connections |

**Example:** If Paper A says "COVID → damages mitochondria" and Paper B says "mitochondrial damage → fatigue", a SQL query for "COVID → fatigue" returns **nothing**. The connection exists but is invisible.

### The Solution

`.causal` pre-computes all transitive chains at storage time:

```
COVID → damages → mitochondria  (explicit, Paper A)
mitochondria → causes → fatigue  (explicit, Paper B)
─────────────────────────────────────────────────────
COVID → indirectly causes → fatigue  (INFERRED, deterministic)
```

**Zero hallucination.** Every inference has full provenance back to source papers.

## Key Features

| Feature | Benefit |
|---------|---------|
| **~30-40x faster queries** | 1.1ms vs 41.5ms (SQLite) - pre-computed inference |
| **50-200% fact amplification** | Weak signals become visible through transitive chains |
| **~60-80% smaller files** | MessagePack + entity deduplication |
| **Zero hallucination** | Pure deterministic logic, full provenance |
| **Edge AI ready** | Small enough for mobile/offline (air-gapped privacy) |
| **Auto-threshold** | Self-adapting fuzzy matching based on entity characteristics |

## Installation

```bash
pip install dotcausal

# With LangChain integration
pip install dotcausal[langchain]
```

## Quick Start

### Python API

```python
from dotcausal import CausalWriter, CausalReader

# Create a knowledge graph
writer = CausalWriter()
writer.add_triplet(
    trigger="SARS-CoV-2",
    mechanism="damages",
    outcome="mitochondria",
    confidence=0.9,
    source="paper_A.pdf"
)
writer.add_triplet(
    trigger="mitochondrial dysfunction",
    mechanism="causes",
    outcome="chronic fatigue",
    confidence=0.85,
    source="paper_B.pdf"
)
writer.save("knowledge.causal")

# Query with inference amplification
reader = CausalReader("knowledge.causal")
stats = reader.get_stats()
print(f"Explicit: {stats['explicit_triplets']}")
print(f"Inferred: {stats['inferred_triplets']}")
print(f"Amplification: {stats['amplification_percent']}%")

# Search
results = reader.search("fatigue")
for r in results:
    tag = "[INFERRED]" if r['is_inferred'] else "[EXPLICIT]"
    print(f"{tag} {r['trigger']} → {r['mechanism']} → {r['outcome']}")
```

### Command Line

```bash
# Show statistics
dotcausal stats knowledge.causal

# Query the graph
dotcausal query knowledge.causal "COVID" --limit 10

# Convert SQLite to .causal
dotcausal convert pipeline.db output.causal

# Export to JSON
dotcausal export knowledge.causal -o output.json

# Validate integrity
dotcausal validate knowledge.causal
```

## The 3-Pass Inference Engine

| Pass | Method | What it finds |
|------|--------|---------------|
| **1** | Exact keyword | A→activates→B + B→activates→C = A→activates→C |
| **2** | Semantic direction | positive×negative = negative chain |
| **3** | Jaro-Winkler fuzzy | "COVID-19" ↔ "SARS-CoV-2" (auto-threshold) |

**Auto-threshold calibration** (v0.2.0+): The fuzzy matching threshold automatically adapts based on entity characteristics:
- Short medical terms → strict (0.88)
- Long scientific phrases → loose (0.72)

## LangChain Integration

Use `.causal` as a drop-in retriever for any LangChain pipeline:

```python
from dotcausal import CausalRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load knowledge graph as retriever
retriever = CausalRetriever.from_file("knowledge.causal", top_k=10)

# Build RAG chain
prompt = ChatPromptTemplate.from_template("""
Answer based on these verified facts:
{context}

Question: {question}
""")

chain = (
    {"context": retriever, "question": lambda x: x}
    | prompt
    | ChatOpenAI(model="gpt-4")
    | StrOutputParser()
)

# Query with zero hallucination grounding
response = chain.invoke("What mechanisms connect COVID to chronic fatigue?")
```

The retriever returns `Document` objects with rich metadata:
- `is_inferred`: Whether the fact was derived or explicit
- `confidence`: Confidence score (0-1)
- `provenance`: Chain of source triplets for inferred facts

## Use Cases

### LLM Grounding (GraphRAG)
```python
# Instead of asking an LLM to find connections (hallucination risk),
# query the deterministic graph and feed results to the LLM
chains = reader.search("drug_X", field="trigger")
# LLM now synthesizes based on verified facts, not guessing
```

### Edge AI / Privacy
The format is compact enough (~3-5MB for thousands of papers) to run **entirely on-device**. No cloud, no data leakage. Perfect for:
- Personal health knowledge graphs
- Offline scientific assistants
- Air-gapped research environments

### Hypothesis Discovery
Weak signals (3 mentions) become visible convergence points (21+ mentions) after inference. This revealed 3 new Long COVID hypothesis candidates that were invisible in SQLite.

## File Format

```
┌─────────────────────────────────────┐
│ HEADER (64 bytes)                   │
│ Magic: "CAUSAL01" | Version | CRC   │
├─────────────────────────────────────┤
│ ENTITIES - Deduplicated dictionary  │
├─────────────────────────────────────┤
│ TRIPLETS - Explicit facts + metadata│
├─────────────────────────────────────┤
│ RULES - Inference rules             │
├─────────────────────────────────────┤
│ CLUSTERS - Semantic groupings       │
├─────────────────────────────────────┤
│ GAPS - Identified knowledge gaps    │
└─────────────────────────────────────┘
```

- **Encoding:** MessagePack (binary) with JSON fallback
- **Integrity:** xxhash64 CRC verification
- **Compression:** ~4.7:1 vs JSON through entity deduplication

## Citation

If you use `.causal` in your research, please cite:

```bibtex
@article{foss2026causal,
  author = {Foss, David Tom},
  title = {The .causal Format: Deterministic Inference for AI-Assisted Hypothesis Amplification},
  journal = {Zenodo},
  year = {2026},
  doi = {10.5281/zenodo.18326222}
}
```

## Links

- **Homepage:** [dotcausal.com](https://dotcausal.com)
- **Whitepaper:** [Zenodo DOI 10.5281/zenodo.18326222](https://doi.org/10.5281/zenodo.18326222)
- **GitHub:** [github.com/DT-Foss/dotcausal](https://github.com/DT-Foss/dotcausal)
- **PyPI:** [pypi.org/project/dotcausal](https://pypi.org/project/dotcausal)

## License

MIT License - see [LICENSE](LICENSE) for details.

---

*"The era of probabilistic guessing is ending; the era of deterministic discovery has begun."*
