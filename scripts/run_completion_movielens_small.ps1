$ErrorActionPreference = "Stop"

$dataDir = "data_rec"
$outDir = "outputs/completion_movielens"
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

# Small sanity run on MovieLens 100K (downloads on first run).
python main_completion.py `
  --dataset ml-100k `
  --download `
  --data_dir $dataDir `
  --save_dir $outDir `
  --device cuda `
  --R 256 --K 42 `
  --epochs 50
