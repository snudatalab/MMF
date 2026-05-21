#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="outputs/expressivity"
mkdir -p "${OUT_DIR}"

python vis_expressivity.py \
  --device cuda \
  --out_dir "${OUT_DIR}" \
  --R_budget 80 \
  --K_list 2,4,8 \
  --mask_mode sin \
  --mmf_epochs 2000 \
  --svdvals_method lowrank \
  --log10
