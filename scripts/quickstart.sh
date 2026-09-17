#!/usr/bin/env bash
# End-to-end on whatever videos you point it at.
#   bash scripts/quickstart.sh /path/to/videos
set -euo pipefail
DIR="${1:-data/clips}"
IDX="${IDX:-.novelty}"
CFG="${CFG:-configs/cpu.yaml}"

python3 -m novelty.cli index "$DIR" --index "$IDX" --config "$CFG"
python3 -m novelty.cli calibrate --index "$IDX" --labels eval/pairs.yaml || \
  python3 -m novelty.cli calibrate --index "$IDX"
python3 -m novelty.cli select --index "$IDX"
python3 -m novelty.cli report --index "$IDX" --out novelty-report.html
echo "open novelty-report.html"
