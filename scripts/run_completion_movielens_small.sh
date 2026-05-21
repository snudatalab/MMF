#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="data_rec"
OUT_DIR="outputs/completion_movielens"
mkdir -p "${OUT_DIR}"

# Small sanity run on MovieLens 100K (downloads on first run).
python main_completion.py \
  --dataset ml-100k \
  --download \
  --data_dir "${DATA_DIR}" \
  --save_dir "${OUT_DIR}" \
  --device cuda \
  --R 256 --K 42 \
  --epochs 50
