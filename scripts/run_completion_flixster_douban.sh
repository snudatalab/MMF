#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="data_rec"
OUT_DIR="outputs/completion_flixster_douban"
mkdir -p "${OUT_DIR}"

# Downloads the preprocessed GC-MC/MGCNN-style splits on first run.
python main_completion.py \
  --dataset flixster \
  --download \
  --data_dir "${DATA_DIR}" \
  --save_dir "${OUT_DIR}" \
  --device cuda \
  --R 256 --K 42 \
  --epochs 100

python main_completion.py \
  --dataset douban \
  --download \
  --data_dir "${DATA_DIR}" \
  --save_dir "${OUT_DIR}" \
  --device cuda \
  --R 256 --K 42 \
  --epochs 100
