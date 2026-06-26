"""
extract_to_db.py — deterministic text -> triplet DB, no LLM.

Reads .txt/.md files (or a directory), runs the rule extractor, passes every
triplet through the 14-step Foss validation gate (reused from the existing
pipeline), and writes survivors into a SQLite DB whose schema matches what
build/build_causal_from_db.py consumes. The .causal file is then one command
away.

Usage:
    python3 extract_to_db.py CORPUS [CORPUS ...] --db out.db [--domain pharma]

CORPUS may be a file or a directory (walked for .txt/.md).
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rule_extractor import extract_from_text

# the 14-step Foss validation gate, bundled locally (extract_causal_triplets_v2
# is copied into this folder; only validate_triplet_v2 is used, no MLX)
try:
    from extract_causal_triplets_v2 import validate_triplet_v2
    _HAS_GATE = True
except Exception:
    _HAS_GATE = False


SCHEMA = """
CREATE TABLE IF NOT EXISTS triplets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger TEXT NOT NULL,
    mechanism TEXT NOT NULL,
    outcome TEXT NOT NULL,
    quantification TEXT,
    confidence TEXT DEFAULT 'medium',
    evidence_sentence TEXT,
    quality_score REAL DEFAULT 0.0,
    source_file TEXT,
    domain TEXT,
    condition TEXT,
    co_causes TEXT,
    effect_size TEXT,
    population TEXT,
    temporal TEXT
);
"""


def _iter_files(paths):
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    if f.endswith((".txt", ".md", ".pdf")):
                        yield os.path.join(root, f)
        elif os.path.isfile(p):
            yield p


# common ligatures PDF fonts emit as single glyphs
_LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
              "ﬅ": "ft", "ﬆ": "st", "—": "-", "–": "-", "’": "'", "“": '"',
              "”": '"', "•": " "}


def _clean_pdf_text(raw: str) -> str:
    """Repair PDF text-extraction artifacts so sentences survive:
    fix ligatures, de-hyphenate line breaks, join wrapped lines, drop reference
    markers and lines that are mostly math/symbols (formulas, not prose)."""
    import re
    t = raw
    for lig, rep in _LIGATURES.items():
        t = t.replace(lig, rep)
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)          # de-hyphenate across lines
    t = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", t)        # drop [1], [3,4] citations
    # drop lines that are mostly non-letters (equations, tables, garbled fonts)
    kept = []
    for line in t.split("\n"):
        letters = sum(c.isalpha() or c.isspace() for c in line)
        if not line.strip() or (len(line) > 3 and letters / len(line) >= 0.7):
            kept.append(line)
    t = "\n".join(kept)
    t = re.sub(r"\n(?=[a-z])", " ", t)              # join lines wrapping mid-sentence
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t


def _pdf_to_text(path: str) -> str:
    """PDF -> text, layout-aware. markitdown handles two-column flow correctly
    (pymupdf's default interleaves columns); pymupdf is the fallback. No LLM."""
    try:
        from markitdown import MarkItDown
        return MarkItDown().convert(path).text_content
    except Exception:
        pass
    try:
        import fitz
        doc = fitz.open(path)
        return "\n".join(page.get_text() for page in doc)
    except Exception as exc:
        print(f"  pdf read failed {os.path.basename(path)}: {exc}")
        return ""


def _read_text(path: str) -> str:
    """Read a corpus file. PDFs are converted to plain text (layout-aware, no
    LLM) and cleaned of line-break/ligature artifacts so sentences survive."""
    if path.lower().endswith(".pdf"):
        return _clean_pdf_text(_pdf_to_text(path))
    return open(path, encoding="utf-8", errors="ignore").read()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpora", nargs="+")
    ap.add_argument("--db", required=True)
    ap.add_argument("--domain", default="")
    ap.add_argument("--no-gate", action="store_true",
                    help="skip the Foss validation gate")
    ap.add_argument("--semantic", action="store_true",
                    help="extract typed semantic relations (is-a/has-a/...) "
                         "instead of causal triplets — for natural speech")
    args = ap.parse_args()

    if args.semantic:
        from semantic_extractor import extract_from_text as _extract
        use_gate = False   # the Foss gate is causal-specific
    else:
        _extract = extract_from_text
        use_gate = _HAS_GATE and not args.no_gate
        if not _HAS_GATE and not args.no_gate:
            print("note: Foss gate unavailable, running ungated")

    con = sqlite3.connect(args.db)
    con.executescript(SCHEMA)

    n_raw = n_kept = n_files = 0
    for path in _iter_files(args.corpora):
        n_files += 1
        text = _read_text(path)
        if not text.strip():
            continue
        src = os.path.basename(path)
        seen_kept: list = []
        for rt in _extract(text, domain=args.domain, source=src):
            n_raw += 1
            d = rt.as_dict()
            qscore = 0.0
            if use_gate:
                res = validate_triplet_v2(d, seen_kept, rt.evidence_sentence)
                if not res.is_valid:
                    continue
                d["confidence"] = res.confidence
                qscore = res.quality_score
            import json as _json
            co = _json.dumps(d["co_causes"]) if d.get("co_causes") else None
            con.execute(
                "INSERT INTO triplets (trigger, mechanism, outcome, "
                "quantification, confidence, evidence_sentence, quality_score, "
                "source_file, domain, condition, co_causes, effect_size, "
                "population, temporal) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (d["trigger"], d["mechanism"], d["outcome"],
                 d["quantification"], d["confidence"], d["evidence_sentence"],
                 qscore, src, d["domain"],
                 d.get("condition"), co, d.get("effect_size"),
                 d.get("population"), d.get("temporal")))
            seen_kept.append(d)
            n_kept += 1
    con.commit()
    con.close()

    gate = "Foss-gated" if use_gate else "ungated"
    print(f"files: {n_files} | extracted: {n_raw} | kept ({gate}): {n_kept} "
          f"| rejected: {n_raw - n_kept}")
    print(f"db written: {args.db}")
    print(f"next: python3 ../build/build_causal_from_db.py --db {args.db} "
          f"-o ../graphs/out.causal")


if __name__ == "__main__":
    main()
