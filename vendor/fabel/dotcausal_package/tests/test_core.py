"""Tests for dotcausal core functionality."""

import pytest
import tempfile
from pathlib import Path

from dotcausal import (
    CausalWriter,
    CausalReader,
    CausalFile,
    jaro_winkler_similarity,
    detect_mechanism_direction,
    TrigramIndex,
    __version__,
)


def test_version():
    """Test version is set."""
    assert __version__ == "0.1.0"


def test_jaro_winkler_exact_match():
    """Test Jaro-Winkler similarity for exact matches."""
    assert jaro_winkler_similarity("hello", "hello") == 1.0
    assert jaro_winkler_similarity("COVID-19", "COVID-19") == 1.0


def test_jaro_winkler_similar():
    """Test Jaro-Winkler for similar strings."""
    sim = jaro_winkler_similarity("COVID-19", "SARS-CoV-2")
    assert 0.5 < sim < 1.0


def test_jaro_winkler_different():
    """Test Jaro-Winkler for very different strings."""
    sim = jaro_winkler_similarity("apple", "orange")
    assert sim < 0.7


def test_direction_detection_positive():
    """Test mechanism direction detection for positive indicators."""
    assert detect_mechanism_direction("activates the pathway") == "positive"
    assert detect_mechanism_direction("promotes growth") == "positive"
    assert detect_mechanism_direction("enhances expression") == "positive"


def test_direction_detection_negative():
    """Test mechanism direction detection for negative indicators."""
    assert detect_mechanism_direction("inhibits the enzyme") == "negative"
    assert detect_mechanism_direction("reduces inflammation") == "negative"
    assert detect_mechanism_direction("blocks the receptor") == "negative"


def test_direction_detection_negative_prefix():
    """Test that negative prefixes flip direction."""
    # "deactivates" should be negative (prefix "de" + "activat")
    assert detect_mechanism_direction("deactivates the gene") == "negative"


def test_trigram_index():
    """Test trigram index for candidate retrieval."""
    index = TrigramIndex()
    index.add_batch(["mitochondrial dysfunction", "mitochondrial damage", "oxidative stress"])

    candidates = index.find_candidates("mitochondrial impairment", min_shared=2)
    assert len(candidates) > 0

    # Should find mitochondrial-related entities
    found_entities = [c[0] for c in candidates]
    assert any("mitochondrial" in e for e in found_entities)


def test_writer_reader_roundtrip():
    """Test writing and reading a .causal file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test.causal"

        # Write
        writer = CausalWriter()
        writer.add_triplet(
            trigger="A",
            mechanism="causes",
            outcome="B",
            confidence=0.9
        )
        writer.add_triplet(
            trigger="B",
            mechanism="causes",
            outcome="C",
            confidence=0.85
        )
        stats = writer.save(filepath)

        assert stats['triplets'] == 2
        assert stats['entities'] == 5  # A, causes, B, causes, C (dedup)
        assert filepath.exists()

        # Read
        reader = CausalReader(filepath)
        explicit = reader.get_all_triplets(include_inferred=False)

        assert len(explicit) == 2
        assert explicit[0]['trigger'] == "A"
        assert explicit[0]['mechanism'] == "causes"
        assert explicit[0]['outcome'] == "B"


def test_integrity_check():
    """Test that corrupted files are detected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test.causal"

        # Create valid file
        writer = CausalWriter()
        writer.add_triplet(trigger="A", mechanism="causes", outcome="B")
        writer.save(filepath)

        # Corrupt the file
        with open(filepath, 'r+b') as f:
            f.seek(100)
            f.write(b'CORRUPTED')

        # Should raise on integrity check
        with pytest.raises(ValueError, match="Integrity check FAILED"):
            CausalReader(filepath, verify_integrity=True)


def test_search():
    """Test search functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test.causal"

        writer = CausalWriter()
        writer.add_triplet(trigger="COVID-19", mechanism="causes", outcome="fatigue")
        writer.add_triplet(trigger="exercise", mechanism="reduces", outcome="fatigue")
        writer.save(filepath)

        reader = CausalReader(filepath)

        results = reader.search("fatigue")
        assert len(results) == 2

        results = reader.search("COVID", field="trigger")
        assert len(results) == 1


def test_inference_transitive():
    """Test that inference generates transitive chains."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test.causal"

        writer = CausalWriter()
        # A causes B, B causes C -> should infer A causes C
        writer.add_triplet(trigger="A", mechanism="causes", outcome="B", confidence=0.9)
        writer.add_triplet(trigger="B", mechanism="causes", outcome="C", confidence=0.9)
        writer.save(filepath)

        reader = CausalReader(filepath)

        explicit = reader.get_all_triplets(include_inferred=False)
        all_facts = reader.get_all_triplets(include_inferred=True)

        assert len(explicit) == 2
        assert len(all_facts) > 2  # Should have inferred facts

        # Check for inferred A -> C
        inferred = [t for t in all_facts if t.get('is_inferred')]
        assert len(inferred) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
