#!/usr/bin/env bash
# Reproduce the ER demo on the bundled toy data (or your own CSV tables).
#
#   bash scripts/run_er.sh [data_dir] [out_dir]
#
# Uses the TF-IDF feature fallback when no PLM cache is present.  For
# paper-level results, pre-compute RoBERTa features into
# <data_dir>/plm/row_features.vec first (see sheet/features.py).
set -euo pipefail

DATA_DIR="${1:-data}"
OUT_DIR="${2:-output}"
PYTHON="${PYTHON:-python}"   # e.g. PYTHON=/path/to/envs/hetea/bin/python
cd "$(dirname "$0")/.."

"$PYTHON" -m sheet.train \
  --data_dir "$DATA_DIR" \
  --out_dir "$OUT_DIR" \
  --num_epochs 100 \
  --hidden_dim 64 \
  --output_dim 256 \
  --num_layers 2 \
  --num_heads 4 \
  --lr 1e-3 \
  --seed 11
