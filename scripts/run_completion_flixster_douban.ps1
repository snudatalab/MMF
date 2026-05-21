$ErrorActionPreference = "Stop"

$dataDir = "data_rec"
$outDir = "outputs/completion_flixster_douban"
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

# Downloads the preprocessed GC-MC/MGCNN-style splits on first run.
python main_completion.py `
  --dataset flixster `
  --download `
  --data_dir $dataDir `
  --save_dir $outDir `
  --device cuda `
  --R 256 --K 42 `
  --epochs 100

python main_completion.py `
  --dataset douban `
  --download `
  --data_dir $dataDir `
  --save_dir $outDir `
  --device cuda `
  --R 256 --K 42 `
  --epochs 100
