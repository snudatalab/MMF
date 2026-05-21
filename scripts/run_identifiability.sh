#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="outputs/identifiability"
mkdir -p "${OUT_DIR}"

python vis_identifiability.py \
  --device cuda \
  --out_dir "${OUT_DIR}" \
  --mode fixed_mask \
  --I 512 --J 512 \
  --R 40 --K 4 \
  --epochs 4000 \
  --mask_mode gaussian \
  --reorder
