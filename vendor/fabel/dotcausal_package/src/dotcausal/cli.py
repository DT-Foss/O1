"""
Command-line interface for dotcausal.

Usage:
    dotcausal stats knowledge.causal
    dotcausal query knowledge.causal "COVID"
    dotcausal convert pipeline.db output.causal
    dotcausal export knowledge.causal -o output.json
    dotcausal validate knowledge.causal
"""

import json
import sys
from pathlib import Path

import click

from . import __version__
from .io import CausalReader, CausalWriter


@click.group()
@click.version_option(version=__version__, prog_name="dotcausal")
def main():
    """
    dotcausal - Binary Knowledge Graph Format with Embedded Inference

    The .causal format provides:
    - 72% smaller files than SQLite
    - 1.9x fact amplification through inference
    - Zero hallucination guarantee

    Documentation: https://dotcausal.com/docs
    """
    pass


@main.command()
@click.argument('file', type=click.Path(exists=True))
def stats(file):
    """Show statistics for a .causal file."""
    try:
        reader = CausalReader(file)
        s = reader.get_stats()

        click.echo()
        click.echo(click.style("=" * 50, fg="blue"))
        click.echo(click.style(f"  DOTCAUSAL FILE: {Path(file).name}", fg="blue", bold=True))
        click.echo(click.style("=" * 50, fg="blue"))
        click.echo()
        click.echo(f"  File Size:      {s['file_size_kb']} KB")
        click.echo(f"  Version:        {s['version']}")
        click.echo(f"  API ID:         {s['api_id']}")
        click.echo()
        click.echo(click.style("  Knowledge Graph:", bold=True))
        click.echo(f"    Entities:     {s['entities']:,}")
        click.echo(f"    Explicit:     {s['explicit_triplets']:,} triplets")
        click.echo(f"    Inferred:     {s['inferred_triplets']:,} triplets")
        click.echo(f"    Total:        {s['total_triplets']:,} triplets")
        click.echo()
        click.echo(click.style(f"    Amplification: {s['amplification_percent']}%", fg="green", bold=True))
        click.echo()
        click.echo(f"  Rules:          {s['rules']}")
        click.echo(f"  Clusters:       {s['clusters']}")
        click.echo(f"  Gaps:           {s['gaps']}")
        click.echo()

    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(1)


@main.command()
@click.argument('file', type=click.Path(exists=True))
@click.argument('query')
@click.option('--field', '-f', default='all', help='Field to search: trigger, mechanism, outcome, or all')
@click.option('--limit', '-n', default=20, help='Maximum results to show')
@click.option('--include-inferred/--explicit-only', default=True, help='Include inferred triplets')
def query(file, query, field, limit, include_inferred):
    """Search triplets in a .causal file."""
    try:
        reader = CausalReader(file)

        # Get triplets
        if include_inferred:
            all_triplets = reader.get_all_triplets(include_inferred=True)
        else:
            all_triplets = reader.get_all_triplets(include_inferred=False)

        # Filter
        query_lower = query.lower()
        results = []
        for t in all_triplets:
            if field == 'all':
                text = f"{t['trigger']} {t['mechanism']} {t['outcome']}"
            else:
                text = t.get(field, '')

            if query_lower in text.lower():
                results.append(t)

        results = results[:limit]

        click.echo()
        click.echo(f"Found {len(results)} results for '{query}':")
        click.echo()

        for i, r in enumerate(results, 1):
            inferred_tag = click.style(" [INFERRED]", fg="yellow") if r.get('is_inferred') else ""
            click.echo(f"[{i}]{inferred_tag}")
            click.echo(f"  {click.style(r['trigger'], fg='cyan')}")
            click.echo(f"    --[{r['mechanism']}]-->")
            click.echo(f"  {click.style(r['outcome'], fg='green')}")
            click.echo(f"  Confidence: {r['confidence']:.2f}")
            if r.get('quantification'):
                click.echo(f"  Quant: {r['quantification']}")
            click.echo()

    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(1)


@main.command()
@click.argument('file', type=click.Path(exists=True))
@click.option('--output', '-o', default='export.json', help='Output JSON file')
@click.option('--include-inferred/--explicit-only', default=True, help='Include inferred triplets')
@click.option('--format', '-f', 'fmt', type=click.Choice(['json', 'jsonl', 'csv']), default='json')
def export(file, output, include_inferred, fmt):
    """Export .causal file to other formats."""
    try:
        reader = CausalReader(file)
        triplets = reader.get_all_triplets(include_inferred=include_inferred)

        if fmt == 'json':
            with open(output, 'w') as f:
                json.dump({
                    'stats': reader.get_stats(),
                    'triplets': triplets
                }, f, indent=2)

        elif fmt == 'jsonl':
            with open(output, 'w') as f:
                for t in triplets:
                    f.write(json.dumps(t) + '\n')

        elif fmt == 'csv':
            import csv
            with open(output, 'w', newline='') as f:
                if triplets:
                    writer = csv.DictWriter(f, fieldnames=['trigger', 'mechanism', 'outcome', 'confidence', 'is_inferred', 'source'])
                    writer.writeheader()
                    for t in triplets:
                        writer.writerow({
                            'trigger': t['trigger'],
                            'mechanism': t['mechanism'],
                            'outcome': t['outcome'],
                            'confidence': t['confidence'],
                            'is_inferred': t['is_inferred'],
                            'source': t.get('source', '')
                        })

        click.echo(f"Exported {len(triplets)} triplets to {output}")

    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(1)


@main.command()
@click.argument('db', type=click.Path(exists=True))
@click.option('--output', '-o', default='converted.causal', help='Output .causal file')
def convert(db, output):
    """Convert SQLite database to .causal format."""
    try:
        import sqlite3

        click.echo(f"Converting {db} to {output}...")

        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row

        # Try to get triplets with source file info
        try:
            rows = conn.execute("""
                SELECT t.*, d.filename as source_file
                FROM triplets t
                LEFT JOIN chunks c ON t.chunk_id = c.id
                LEFT JOIN documents d ON c.doc_id = d.id
            """).fetchall()
        except:
            # Fallback for simpler schemas
            rows = conn.execute("SELECT * FROM triplets").fetchall()

        writer = CausalWriter(api_id="convert")

        conf_map = {'high': 0.9, 'medium': 0.7, 'low': 0.5}

        for row in rows:
            row_dict = dict(row)

            conf = row_dict.get('confidence', 'medium')
            if isinstance(conf, str):
                conf_val = conf_map.get(conf.lower(), 0.7)
            else:
                conf_val = float(conf) if conf else 0.7

            writer.add_triplet(
                trigger=row_dict.get('trigger', ''),
                mechanism=row_dict.get('mechanism', ''),
                outcome=row_dict.get('outcome', ''),
                confidence=conf_val,
                source=row_dict.get('source_file', ''),
                quantification=row_dict.get('quantification', ''),
                evidence=row_dict.get('evidence_sentence', ''),
                domain=row_dict.get('domain', ''),
                quality_score=row_dict.get('quality_score', 0.0)
            )

        conn.close()

        stats = writer.save(output)

        click.echo()
        click.echo(click.style("Conversion complete!", fg="green", bold=True))
        click.echo(f"  Triplets: {stats['triplets']:,}")
        click.echo(f"  Entities: {stats['entities']:,}")
        click.echo(f"  File size: {stats['file_size_kb']} KB")
        click.echo()

    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(1)


@main.command()
@click.argument('file', type=click.Path(exists=True))
def validate(file):
    """Validate a .causal file."""
    try:
        click.echo(f"Validating {file}...")

        # This will raise if validation fails
        reader = CausalReader(file, verify_integrity=True)
        stats = reader.get_stats()

        click.echo()
        click.echo(click.style("VALID", fg="green", bold=True))
        click.echo(f"  Magic:    CAUSAL01")
        click.echo(f"  Version:  {stats['version']}")
        click.echo(f"  CRC:      OK")
        click.echo(f"  Triplets: {stats['explicit_triplets']:,}")
        click.echo(f"  Entities: {stats['entities']:,}")
        click.echo()

    except ValueError as e:
        click.echo()
        click.echo(click.style("INVALID", fg="red", bold=True))
        click.echo(f"  Error: {e}")
        click.echo()
        sys.exit(1)

    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
