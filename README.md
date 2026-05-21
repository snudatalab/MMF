# Masked Mixture Factorization (MMF)

This repository contains standalone Python scripts used to run MMF experiments:

- **Reconstruction (fully observed synthetic matrix)**: `main_reconstruction.py`
- **Matrix completion (explicit ratings)**: `main_completion.py`
- **Expressivity via singular-value spectrum**: `vis_expressivity.py`
- **Identifiability / factor stability visualization**: `vis_identifiability.py`

---

## 1) Setup

### Requirements

- Python **3.9+**
- Recommended: a CUDA-enabled GPU for the larger runs

Install dependencies:

```bash
python -m pip install -r requirements.txt
```
---

## 2) Quickstart

These are **tiny** runs meant to verify everything works end-to-end.

- Bash (Linux/macOS/Git-Bash):
```bash
bash scripts/run_all_smoke.sh
```

- PowerShell (Windows):
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_all_smoke.ps1
```

Outputs are written under `outputs/`.

---

## 3) Reconstruction experiment (synthetic, fully observed)

The reconstruction script benchmarks MMF against common baselines (e.g., truncated SVD / MF variants) on a fully observed matrix and logs errors and timing.

### Example runs

- Budget sweep (vary rank budget, fixed `K_ratio`):
```bash
bash scripts/run_reconstruction_budget_sweep.sh
```

Or directly:

```bash
python main_reconstruction.py --device cuda --I 256 --J 256   --R_list 16,32,48,64,80 --K_ratio 0.75 --out_dir outputs/reconstruction_budget
```

---

## 4) Matrix completion (explicit ratings)

This script runs MF vs. MMF on explicit-rating matrix completion and saves per-seed + aggregate JSON results.

Supported datasets include MovieLens and (preprocessed) Flixster/Douban.

### MovieLens

```bash
bash scripts/run_completion_movielens_small.sh
```

### Flixster / Douban

```bash
bash scripts/run_completion_flixster_douban.sh
```
---

## 5) Expressivity (singular value spectrum)

The expressivity script reconstructs synthetic matrices with SVD vs. MMF under a similar parameter budget and plots singular-value spectra.

Run:

```bash
bash scripts/run_expressivity.sh
```

Tip: use `--svdvals_method lowrank` for speed on large matrices.

---

## 6) Identifiability / factor stability visualization

This script trains two independent runs and compares base factors via an absolute cosine-similarity heatmap.

Run:

```bash
bash scripts/run_identifiability.sh
```

If you pass `--reorder`, it uses the Hungarian algorithm (requires SciPy) to correct permutation ambiguity in the visualization.

---

