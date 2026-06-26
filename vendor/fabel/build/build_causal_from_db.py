#!/usr/bin/env python3
"""
Build .causal from SQLite Database
===================================

Workflow:
1. Read ALL explicit triplets from pipeline.db
2. Convert to .causal format
3. Run inference ONCE on all triplets together
4. Apply quality filters
5. Output final .causal file

Usage:
    python3 build_causal_from_db.py
    python3 build_causal_from_db.py --db custom.db --output custom.causal
    python3 build_causal_from_db.py --skip-inference  # Only convert, no inference

Author: Sovereign Pipeline Team
Date: 2026-01-22
"""

import sys
import time
import sqlite3
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, '.')
from causal_format import CausalWriter, CausalReader, INFERENCE_CONFIG

# deterministic entity canonicalization (collapses variants to one graph node,
# drops sentence-debris hubs). The raw text stays in the DB; only the GRAPH
# collapses. Lives in extract/; import it whichever way the build is launched.
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                 "..", "extract"))
try:
    from normalize import canonical as _canonical, is_entity as _is_entity
    _HAS_NORM = True
except Exception:
    _HAS_NORM = False
    def _canonical(e):       # fallbacks so references stay bound when absent
        return (e or "").strip().lower()
    def _is_entity(c):
        return bool(c)


# =============================================================================
# QUALITY FILTERS
# =============================================================================

CONTRADICTION_PATTERNS = [
    (r'(?:HBOT|antioxidant|protective|therapy|treatment)', r'promotes?.*(?:injury|damage|harm|death)'),
]

# NOTE: "indirectly linked to" is NOT filtered anymore!
# Experiment showed these are actually honest (admit uncertainty)
# while "directional" ones can be semantically wrong.
# See EXPERIMENT_LOG.md for details.
VAGUE_CONCLUSIONS = []  # Was: ['indirectly linked to', 'associated with', 'related to']


def quality_check_inference(inf: Dict) -> Tuple[bool, str]:
    """
    Check if an inferred triplet passes quality filters.

    Filters (after 2026-01-22 experiment):
    1. Confidence >= 0.35 (main quality gate)
    2. Contradiction patterns (semantic errors)
    3. Self-referential (A → A)

    NOT filtered: "indirectly linked to" (these are honest about uncertainty)
    """
    trigger = inf.get('trigger', '').lower()
    mechanism = inf.get('mechanism', '').lower()
    outcome = inf.get('outcome', '').lower()
    confidence = inf.get('confidence', 0)

    # 1. Confidence threshold
    if confidence < 0.35:
        return False, f"Low confidence: {confidence:.2f}"

    # 2. Check for contradiction patterns
    for trigger_pattern, outcome_pattern in CONTRADICTION_PATTERNS:
        if re.search(trigger_pattern, trigger, re.I) and re.search(outcome_pattern, f"{mechanism} {outcome}", re.I):
            return False, "Contradiction pattern"

    # 3. Self-referential check
    if trigger[:20].lower() == outcome[:20].lower():
        return False, "Self-referential"

    # 4. Vague conclusion filter (DISABLED after experiment)
    # Neutral "linked to" is actually more honest than wrong directional claims
    for vague in VAGUE_CONCLUSIONS:
        if vague in mechanism:
            return False, f"Vague: {vague}"

    return True, ""


# =============================================================================
# MAIN BUILD FUNCTION
# =============================================================================

def build_causal_from_db(
    db_path: str = "pipeline.db",
    output_path: str = None,
    run_inference: bool = True,
    min_confidence: float = 0.5,
    normalize_entities: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Build .causal file from SQLite database.

    Args:
        db_path: Path to pipeline.db
        output_path: Output .causal path (default: pipeline_YYYYMMDD.causal)
        run_inference: Run inference after building (default: True)
        min_confidence: Minimum confidence for explicit triplets
        verbose: Print progress

    Returns:
        Stats dict
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    # Default output name
    if output_path is None:
        date_str = datetime.now().strftime("%Y%m%d")
        output_path = f"pipeline_{date_str}.causal"

    if verbose:
        print("=" * 70)
        print("BUILD .CAUSAL FROM DATABASE")
        print("=" * 70)
        print(f"Input:  {db_path}")
        print(f"Output: {output_path}")
        print(f"Inference: {'Yes' if run_inference else 'No'}")
        print("=" * 70)

    start_time = time.time()

    # ==========================================================================
    # STEP 1: Load triplets from SQLite
    # ==========================================================================
    if verbose:
        print("\n[1/4] Loading triplets from database...")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Check schema - handle both old and new formats
    cursor = conn.execute("PRAGMA table_info(triplets)")
    columns = {row[1] for row in cursor.fetchall()}

    # Build query based on available columns
    base_cols = ['id', 'trigger', 'mechanism', 'outcome', 'confidence', 'quantification', 'evidence_sentence']
    optional_cols = ['source_file', 'domain', 'quality_score', 'pmcid',
                     'condition', 'co_causes', 'effect_size', 'population',
                     'temporal']

    select_cols = []
    for col in base_cols:
        if col in columns:
            select_cols.append(col)
        else:
            select_cols.append(f"NULL as {col}")

    for col in optional_cols:
        if col in columns:
            select_cols.append(col)
        else:
            select_cols.append(f"'' as {col}")

    query = f"SELECT {', '.join(select_cols)} FROM triplets"
    rows = conn.execute(query).fetchall()
    conn.close()

    if verbose:
        print(f"      Loaded {len(rows):,} triplets from database")

    # ==========================================================================
    # STEP 2: Write to .causal format
    # ==========================================================================
    if verbose:
        print("\n[2/4] Writing explicit triplets to .causal...")

    writer = CausalWriter(api_id="db_build")
    skipped = 0
    norm_dropped = 0          # edges dropped: an endpoint was sentence-debris
    norm_selfloop = 0         # edges dropped: collapsed to A→A after canonicalize
    do_norm = normalize_entities and _HAS_NORM

    for row in rows:
        # Parse confidence
        conf = row['confidence']
        if isinstance(conf, str):
            conf_map = {'high': 0.9, 'medium': 0.7, 'low': 0.5}
            conf = conf_map.get(conf.lower(), 0.7)
        else:
            conf = float(conf) if conf else 0.7

        # Skip low confidence
        if conf < min_confidence:
            skipped += 1
            continue

        # Canonicalize endpoints to graph node keys. Raw text survives in the
        # evidence sentence; only the node identity collapses, so variants meet
        # at one node (the measured 23× chain lift). Junk endpoints are dropped.
        node_trigger = row['trigger'] or ''
        node_outcome = row['outcome'] or ''
        if do_norm:
            ct = _canonical(node_trigger)
            co_ = _canonical(node_outcome)
            if not _is_entity(ct) or not _is_entity(co_):
                norm_dropped += 1
                continue
            if ct == co_:
                norm_selfloop += 1
                continue
            node_trigger, node_outcome = ct, co_

        # v1.1: carry the typed relation fields through as attrs
        import json as _json
        keys = row.keys()
        co = None
        if 'co_causes' in keys and row['co_causes']:
            try:
                co = _json.loads(row['co_causes'])
            except Exception:
                co = None
        attrs = {
            'condition': row['condition'] if 'condition' in keys else None,
            'co_causes': co,
            'effect_size': row['effect_size'] if 'effect_size' in keys else None,
            'population': row['population'] if 'population' in keys else None,
            'temporal': row['temporal'] if 'temporal' in keys else None,
        }
        # the canonical node goes into trigger/outcome; the verbatim sentence —
        # and the original surface forms — stay in the evidence so nothing is lost
        evidence = row['evidence_sentence'] or ''
        if do_norm and (node_trigger != (row['trigger'] or '').strip().lower()
                        or node_outcome != (row['outcome'] or '').strip().lower()):
            surface = f"[{row['trigger']} → {row['outcome']}] {evidence}".strip()
            evidence = surface
        writer.add_triplet(
            trigger=node_trigger,
            mechanism=row['mechanism'] or '',
            outcome=node_outcome,
            confidence=conf,
            source=row['source_file'] if 'source_file' in row.keys() else '',
            pmcid=row['pmcid'] if 'pmcid' in row.keys() else '',
            quantification=row['quantification'] or '',
            evidence=evidence,
            domain=row['domain'] if 'domain' in row.keys() else '',
            quality_score=float(row['quality_score']) if row['quality_score'] else 0.0,
            attrs=attrs
        )

    # Save explicit-only .causal
    stats = writer.save(output_path)

    if verbose:
        print(f"      Written {stats['triplets']:,} explicit triplets")
        print(f"      Skipped {skipped:,} low-confidence triplets")
        if do_norm:
            print(f"      Normalized entities ON: dropped {norm_dropped:,} junk-"
                  f"endpoint edges, {norm_selfloop:,} self-loops after collapse")
        elif normalize_entities and not _HAS_NORM:
            print("      (normalize.py not importable — entities NOT canonicalized)")
        print(f"      File size: {stats['file_size_kb']:.1f} KB")

    # ==========================================================================
    # STEP 3: Run inference (optional)
    # ==========================================================================
    if run_inference:
        if verbose:
            print("\n[3/4] Running inference on explicit triplets...")
            print(f"      Config: {INFERENCE_CONFIG}")

        reader = CausalReader(output_path, verify_integrity=False)

        # This runs the inference engine
        inf_start = time.time()
        all_triplets = reader.get_all_triplets(include_inferred=True)
        inf_time = time.time() - inf_start

        explicit_count = stats['triplets']
        inferred_raw = [t for t in all_triplets if t.get('is_inferred', False)]

        if verbose:
            print(f"      Inference time: {inf_time:.1f}s")
            print(f"      Raw inferred: {len(inferred_raw):,}")

        # ==========================================================================
        # STEP 4: Apply quality filters and rebuild
        # ==========================================================================
        if verbose:
            print("\n[4/4] Applying quality filters and rebuilding...")

        # Filter inferred triplets
        quality_passed = []
        rejections = defaultdict(int)

        for inf in inferred_raw:
            passed, reason = quality_check_inference(inf)
            if passed:
                quality_passed.append(inf)
            else:
                rejections[reason] += 1

        if verbose:
            print(f"      Quality passed: {len(quality_passed):,}")
            print(f"      Rejected: {len(inferred_raw) - len(quality_passed):,}")
            for reason, count in sorted(rejections.items(), key=lambda x: -x[1])[:5]:
                print(f"        - {reason}: {count}")

        # Rebuild with explicit + quality-filtered inferred
        final_writer = CausalWriter(api_id="final")

        # Add explicit triplets (carry typed attrs through the rebuild too)
        for t in all_triplets:
            if not t.get('is_inferred', False):
                final_writer.add_triplet(
                    trigger=t['trigger'],
                    mechanism=t['mechanism'],
                    outcome=t['outcome'],
                    confidence=t['confidence'],
                    source=t.get('source', ''),
                    pmcid=t.get('pmcid', ''),
                    quantification=t.get('quantification', ''),
                    evidence=t.get('evidence', ''),
                    domain=t.get('domain', ''),
                    quality_score=t.get('quality_score', 0.0),
                    attrs=t.get('attrs')
                )

        # Add quality-filtered inferred triplets, PERSISTED as inferred so a
        # later reader uses them as-is instead of re-running the 3-pass engine.
        for t in quality_passed:
            final_writer.add_triplet(
                trigger=t['trigger'],
                mechanism=t['mechanism'],
                outcome=t['outcome'],
                confidence=t['confidence'],
                source=t.get('source', 'INFERRED'),  # Keep source marker
                evidence=t.get('evidence', ''),
                domain=t.get('domain', ''),
                is_inferred=True
            )

        final_stats = final_writer.save(output_path)

        stats['inferred_raw'] = len(inferred_raw)
        stats['inferred_filtered'] = len(quality_passed)
        stats['final_triplets'] = final_stats['triplets']
        stats['inference_time'] = inf_time

        # ==========================================================================
        # STEP 5: Export Knowledge Gaps (what we DON'T know)
        # ==========================================================================
        if verbose:
            print("\n[5/5] Exporting knowledge gaps...")

        # Find "indirectly linked to" = uncertain direction = GAPS
        knowledge_gaps = [t for t in quality_passed if 'indirectly linked to' in t.get('mechanism', '')]

        # Group by outcome for convergence analysis
        from collections import defaultdict as dd
        gaps_by_outcome = dd(list)
        for g in knowledge_gaps:
            outcome_key = g['outcome'][:80].lower().strip()
            gaps_by_outcome[outcome_key].append(g)

        # Build gaps export
        gaps_export = {
            'generated': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'source': str(output_path),
            'total_uncertain': len(knowledge_gaps),
            'convergence_points': len(gaps_by_outcome),
            'gaps': []
        }

        # Top 100 convergence points (most uncertain paths → same outcome)
        for outcome, triggers in sorted(gaps_by_outcome.items(), key=lambda x: -len(x[1]))[:100]:
            trigger_samples = list(set(t['trigger'][:50] for t in triggers[:5]))
            search_terms = set()
            for t in triggers[:5]:
                search_terms.update(w for w in t['trigger'].lower().split() if len(w) > 3)
                search_terms.update(w for w in outcome.split() if len(w) > 3)

            gaps_export['gaps'].append({
                'outcome': outcome,
                'uncertain_triggers': len(triggers),
                'priority': min(1.0, len(triggers) / 50),
                'trigger_samples': trigger_samples,
                'search_terms': list(search_terms)[:10],
                'pubmed_query': f"{' '.join(list(search_terms)[:4])} mechanism pathway"
            })

        # Save gaps file
        gaps_path = str(output_path).replace('.causal', '_GAPS.json')
        with open(gaps_path, 'w') as f:
            import json
            json.dump(gaps_export, f, indent=2)

        stats['gaps_file'] = gaps_path
        stats['knowledge_gaps'] = len(knowledge_gaps)

        if verbose:
            print(f"      Knowledge gaps: {len(knowledge_gaps):,}")
            print(f"      Convergence points: {len(gaps_by_outcome):,}")
            print(f"      Exported to: {gaps_path}")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    total_time = time.time() - start_time

    if verbose:
        print("\n" + "=" * 70)
        print("BUILD COMPLETE")
        print("=" * 70)
        print(f"Output: {output_path}")
        print(f"Explicit triplets: {stats['triplets']:,}")
        if run_inference:
            print(f"Inferred (raw): {stats.get('inferred_raw', 0):,}")
            print(f"Inferred (filtered): {stats.get('inferred_filtered', 0):,}")
            print(f"Final total: {stats.get('final_triplets', stats['triplets']):,}")
            amplification = stats.get('inferred_filtered', 0) / max(stats['triplets'], 1) * 100
            print(f"Amplification: +{amplification:.1f}%")
        print(f"Total time: {total_time:.1f}s")
        print("=" * 70)

    stats['total_time'] = total_time
    return stats


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Build .causal from SQLite database")
    parser.add_argument('--db', default='pipeline.db', help='Input database path')
    parser.add_argument('--output', '-o', default=None, help='Output .causal path')
    parser.add_argument('--skip-inference', action='store_true', help='Skip inference step')
    parser.add_argument('--min-confidence', type=float, default=0.5, help='Min confidence threshold')
    parser.add_argument('--no-normalize', action='store_true',
                        help='Disable entity canonicalization (keep raw surface '
                             'forms as graph nodes — for A/B comparison)')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')

    args = parser.parse_args()

    stats = build_causal_from_db(
        db_path=args.db,
        output_path=args.output,
        run_inference=not args.skip_inference,
        min_confidence=args.min_confidence,
        normalize_entities=not args.no_normalize,
        verbose=not args.quiet
    )

    return 0 if stats else 1


if __name__ == '__main__':
    sys.exit(main())
