$ErrorActionPreference = "Stop"

$outDir = "outputs/expressivity"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

python vis_expressivity.py `
  --device cuda `
  --out_dir $outDir `
  --R_budget 80 `
  --K_list 2,4,8 `
  --mask_mode sin `
  --mmf_epochs 2000 `
  --svdvals_method lowrank `
  --log10
