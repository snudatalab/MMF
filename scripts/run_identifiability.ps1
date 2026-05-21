$ErrorActionPreference = "Stop"

$outDir = "outputs/identifiability"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

python vis_identifiability.py `
  --device cuda `
  --out_dir $outDir `
  --mode fixed_mask `
  --I 512 --J 512 `
  --R 40 --K 4 `
  --epochs 4000 `
  --mask_mode gaussian `
  --reorder
