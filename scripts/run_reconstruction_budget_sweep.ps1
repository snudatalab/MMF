$ErrorActionPreference = "Stop"

$outDir = "outputs/reconstruction_budget"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

python main_reconstruction.py `
  --device cuda `
  --I 256 --J 256 `
  --R_list 16,32,48,64,80 `
  --K_ratio 0.75 `
  --out_dir $outDir
