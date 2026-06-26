#!/usr/bin/env bash
# run_pipeline.sh — text -> .causal -> talk, end to end, no LLM.
#
#   ./run_pipeline.sh CORPUS [DOMAIN]
#
# CORPUS: a .txt/.md file or a directory of them.
# DOMAIN: optional tag stored on every triplet (e.g. pharma, physics).
set -euo pipefail
cd "$(dirname "$0")"

CORPUS="${1:?usage: run_pipeline.sh CORPUS [DOMAIN]}"
DOMAIN="${2:-}"
NAME="$(basename "${CORPUS%.*}")"
DB="graphs/${NAME}.db"
CAUSAL="graphs/${NAME}.causal"

echo "── 1/3  extract (deterministic, Foss-gated) ──────────────────"
python3 extract/extract_to_db.py "$CORPUS" --db "$DB" --domain "$DOMAIN"

echo "── 2/3  build .causal (embedded inference) ───────────────────"
python3 build/build_causal_from_db.py --db "$DB" -o "$CAUSAL"

echo "── 3/3  ready ────────────────────────────────────────────────"
echo "talk to it:  python3 fabel.py $CAUSAL"
