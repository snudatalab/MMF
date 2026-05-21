#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="outputs/reconstruction_budget"
mkdir -p "${OUT_DIR}"

python main_reconstruction.py \
  --device cuda \
  --I 256 --J 256 \
  --R_list 16,32,48,64,80 \
  --K_ratio 0.75 \
  --out_dir "${OUT_DIR}"
