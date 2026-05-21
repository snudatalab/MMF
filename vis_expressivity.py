"""
vis_expressivity.py

Visualize the singular value spectrum of reconstructed matrices to illustrate the
"effective-rank expansion" that MMF can exhibit under a comparable parameter budget.

This script:
  1) creates synthetic target matrices (dense random S_10, block-diagonal H_6),
  2) reconstructs them using:
       - truncated SVD with rank = R_budget (optimal rank-R approximation),
       - MMF under a roughly matched parameter budget,
  3) plots the singular values of the reconstructions as a heatmap.

Example:
  python vis_expressivity.py --device cuda --out_dir out_express --R_budget 80 --K_list 2,4,8

Notes:
  - Full SVD on large matrices can be expensive. Use --svdvals_method lowrank for speed.
  - The visualization is meant for qualitative comparison (not a rigorous proof).
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib

# Use a non-interactive backend by default (safe for servers/CI).
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def set_seed(seed: int) -> None:
    """Set RNG seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_fro(X: torch.Tensor) -> torch.Tensor:
    """Normalize a matrix by its Frobenius norm."""
    return X / torch.linalg.norm(X).clamp_min(1e-12)


def sample_matrix(I: int, J: int, dist: Literal["normal", "uniform"] = "normal",
                  scale: float = 1.0, device: str = "cpu") -> torch.Tensor:
    if dist == "normal":
        return scale * torch.randn(I, J, device=device)
    if dist == "uniform":
        return scale * (2.0 * torch.rand(I, J, device=device) - 1.0)
    raise ValueError(f"Unknown dist: {dist}")


def split_sizes(total: int, num_blocks: int) -> List[int]:
    base = total // num_blocks
    rem = total % num_blocks
    return [base + (1 if b < rem else 0) for b in range(num_blocks)]


def make_random_X(I: int, J: int, seed: int, device: str) -> torch.Tensor:
    set_seed(seed)
    return normalize_fro(sample_matrix(I, J, dist="normal", scale=1.0, device=device))


def make_block_diag_X(I: int, J: int, num_blocks: int, seed: int, device: str,
                      offblock_noise_scale: float = 0.01) -> torch.Tensor:
    """Block-diagonal synthetic matrix with small off-block noise."""
    set_seed(seed)
    X = sample_matrix(I, J, dist="normal", scale=offblock_noise_scale, device=device)

    row_sizes = split_sizes(I, num_blocks)
    col_sizes = split_sizes(J, num_blocks)

    r0, c0 = 0, 0
    for b in range(num_blocks):
        r1 = r0 + row_sizes[b]
        c1 = c0 + col_sizes[b]
        X[r0:r1, c0:c1] = sample_matrix(row_sizes[b], col_sizes[b], dist="normal", scale=1.0, device=device)
        r0, c0 = r1, c1

    return normalize_fro(X)


def make_mask_fn(mode: Literal["sin", "cos", "gaussian"]) -> callable:
    if mode == "sin":
        return lambda d: torch.sin(d)
    if mode == "cos":
        return lambda d: torch.cos(d)
    if mode == "gaussian":
        return lambda d: torch.exp(-d ** 2 / 2)
    raise ValueError(f"Unknown mask mode: {mode}")


class MMFReconstruction(nn.Module):
    """A minimal MMF-style reconstruction model for spectrum visualization.

    This is intentionally lightweight and self-contained (not the full MMF used in completion).
    """

    def __init__(self, I: int, J: int, R: int, K: int, device: str,
                 mask_mode: Literal["sin", "cos", "gaussian"] = "sin"):
        super().__init__()
        self.I, self.J, self.R, self.K = I, J, R, K
        self.mask_scale = 1.0 / K
        self.mask_scale_square = self.mask_scale ** 2
        self.half = float(R) / 2.0

        init_scale = 0.02
        self.A = nn.Parameter(init_scale * torch.randn(I, R, device=device).unsqueeze(1))  # (I,1,R)
        self.B = nn.Parameter(init_scale * torch.randn(J, R, device=device).unsqueeze(1))  # (J,1,R)

        # Learnable shifts in [0,2) at init; gradients can move them.
        self.uA = nn.Parameter(torch.zeros(I, K, device=device).unsqueeze(-1))
        self.uB = nn.Parameter(torch.zeros(J, K, device=device).unsqueeze(-1))

        pos = torch.arange(R, device=device).view(1, 1, R)                    # (1,1,R)
        omega = torch.linspace(1.0 / K, 1.0, steps=K, device=device).view(1, K, 1)  # (1,K,1)
        self.register_buffer("template", pos * omega, persistent=False)       # (1,K,R)

        self._mask_fn = make_mask_fn(mask_mode)

    def forward(self) -> torch.Tensor:
        deltaA = self.template - self.half * self.uA  # (I,K,R)
        deltaB = self.template - self.half * self.uB  # (J,K,R)
        mA = self._mask_fn(deltaA)
        mB = self._mask_fn(deltaB)

        A_cat = (self.A * mA).reshape(self.I, self.K * self.R)  # (I, K*R)
        B_cat = (self.B * mB).reshape(self.J, self.K * self.R)  # (J, K*R)
        return self.mask_scale_square * (A_cat @ B_cat.T)       # (I,J)


@torch.no_grad()
def svd_reconstruct(X: torch.Tensor, rank: int) -> torch.Tensor:
    """Optimal rank-`rank` approximation under Frobenius norm (truncated SVD)."""
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    r = min(rank, S.numel())
    return (U[:, :r] * S[:r]) @ Vh[:r, :]


@torch.no_grad()
def singular_values(
    X: torch.Tensor,
    top_k: int,
    method: Literal["full", "lowrank"] = "full",
    svd_device: Literal["cpu", "same"] = "cpu",
    oversampling: int = 10,
) -> np.ndarray:
    """Return top-k singular values of X.

    Args:
      method:
        - "full": compute all singular values via torch.linalg.svdvals.
        - "lowrank": compute an approximate top-k spectrum via torch.svd_lowrank.
      svd_device:
        - "cpu": move X to CPU for SVD (more stable, often faster for medium sizes).
        - "same": keep X on the current device.
    """
    if svd_device == "cpu":
        X = X.detach().float().cpu()
    else:
        X = X.detach().float()

    if method == "full":
        S = torch.linalg.svdvals(X)
        return S[:top_k].cpu().numpy()

    # Approximate top-k with randomized low-rank SVD.
    q = min(top_k + oversampling, min(X.shape))
    U, S, V = torch.svd_lowrank(X, q=q, niter=2)
    S = S.sort(descending=True).values
    return S[:top_k].cpu().numpy()


def train_mmf_reconstruction(
    X_target: torch.Tensor,
    R_budget: int,
    K: int,
    device: str,
    epochs: int,
    lr: float,
    mask_mode: Literal["sin", "cos", "gaussian"],
) -> torch.Tensor:
    """Train a minimal MMF reconstruction model and return the final reconstruction."""
    I, J = X_target.shape

    # Very simple budget matching: keep (I+J) * (K*R_base) roughly comparable to (I+J)*R_budget.
    # This heuristic is only for plotting; adjust if you prefer a different matching rule.
    R_base = max(1, R_budget - K)

    model = MMFReconstruction(I, J, R=R_base, K=K, device=device, mask_mode=mask_mode).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        X_hat = model()
        loss = torch.mean((X_hat - X_target) ** 2)
        loss.backward()
        opt.step()

    with torch.no_grad():
        return model().detach()


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def plot_heatmap(
    matrix: np.ndarray,
    row_labels: List[str],
    R_budget: int,
    out_path: str,
    x_tick_step: int = 40,
    use_log10: bool = False,
) -> None:
    """Plot a heatmap of singular values (rows: methods, columns: indices)."""
    data = np.log10(matrix + 1e-12) if use_log10 else matrix

    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(data, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("Method")

    # X ticks
    if x_tick_step > 0:
        xticks = np.arange(0, data.shape[1], x_tick_step)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks])

    # Rank budget indicator
    ax.axvline(x=R_budget, linestyle="--", linewidth=2)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log10(singular value)" if use_log10 else "singular value")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def run_experiment(
    name: str,
    X: torch.Tensor,
    R_budget: int,
    K_list: Iterable[int],
    device: str,
    out_dir: str,
    mmf_epochs: int,
    mmf_lr: float,
    mask_mode: Literal["sin", "cos", "gaussian"],
    svdvals_method: Literal["full", "lowrank"],
    svdvals_k_mult: float,
    svd_device: Literal["cpu", "same"],
    use_log10: bool,
) -> None:
    print(f"[{name}] running spectrum analysis (R_budget={R_budget})")

    # Determine how many singular values to plot.
    top_k = int(max(1, math.ceil(R_budget * svdvals_k_mult)))

    spectra: List[np.ndarray] = []
    labels: List[str] = []

    # 1) SVD baseline
    X_svd = svd_reconstruct(X, rank=R_budget)
    spectra.append(singular_values(X_svd, top_k=top_k, method=svdvals_method, svd_device=svd_device))
    labels.append("SVD (rank=R_budget)")

    # 2) MMF variants
    for K in K_list:
        X_mmf = train_mmf_reconstruction(
            X_target=X,
            R_budget=R_budget,
            K=K,
            device=device,
            epochs=mmf_epochs,
            lr=mmf_lr,
            mask_mode=mask_mode,
        )
        spectra.append(singular_values(X_mmf, top_k=top_k, method=svdvals_method, svd_device=svd_device))
        labels.append(f"MMF (K={K})")

    max_len = max(len(s) for s in spectra)
    mat = np.zeros((len(spectra), max_len), dtype=np.float64)
    for i, s in enumerate(spectra):
        mat[i, : len(s)] = s

    out_path = os.path.join(out_dir, f"spectrum_{name}.pdf")
    plot_heatmap(mat, labels, R_budget=R_budget, out_path=out_path, use_log10=use_log10)
    print(f"[Saved] {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", type=str, default="out_express")
    p.add_argument("--seed", type=int, default=0)

    # Budget and sweep
    p.add_argument("--R_budget", type=int, default=80, help="Equivalent rank budget for the SVD baseline.")
    p.add_argument("--K_list", type=str, default="2,3,4,5,6,7,8,9", help="Comma-separated list of K values for MMF.")
    p.add_argument("--mask_mode", type=str, default="sin", choices=["sin", "cos", "gaussian"])

    # MMF training
    p.add_argument("--mmf_epochs", type=int, default=2000)
    p.add_argument("--mmf_lr", type=float, default=0.02)

    # SVD spectrum extraction
    p.add_argument("--svdvals_method", type=str, default="full", choices=["full", "lowrank"])
    p.add_argument("--svdvals_k_mult", type=float, default=8.0, help="Plot top_k = ceil(R_budget * k_mult) values.")
    p.add_argument("--svd_device", type=str, default="cpu", choices=["cpu", "same"])

    # Datasets
    p.add_argument("--S10_size", type=int, default=1024, help="Matrix size for S_10 (dense random).")
    p.add_argument("--H6_size", type=int, default=1080, help="Matrix size for H_6 (block diagonal).")
    p.add_argument("--H6_blocks", type=int, default=6)

    # Plotting
    p.add_argument("--log10", action="store_true", help="Plot log10(singular values) for better tail visibility.")
    args = p.parse_args()

    ensure_dir(args.out_dir)
    set_seed(args.seed)

    K_list = parse_int_list(args.K_list)
    device = args.device

    print("[Data] generating S_10 ...")
    X_s10 = make_random_X(args.S10_size, args.S10_size, seed=args.seed, device=device)

    print("[Data] generating H_6 ...")
    X_h6 = make_block_diag_X(args.H6_size, args.H6_size, num_blocks=args.H6_blocks,
                            seed=args.seed, device=device)

    run_experiment(
        name="S_10",
        X=X_s10,
        R_budget=args.R_budget,
        K_list=K_list,
        device=device,
        out_dir=args.out_dir,
        mmf_epochs=args.mmf_epochs,
        mmf_lr=args.mmf_lr,
        mask_mode=args.mask_mode,
        svdvals_method=args.svdvals_method,
        svdvals_k_mult=args.svdvals_k_mult,
        svd_device=args.svd_device,
        use_log10=args.log10,
    )

    run_experiment(
        name="H_6",
        X=X_h6,
        R_budget=args.R_budget,
        K_list=K_list,
        device=device,
        out_dir=args.out_dir,
        mmf_epochs=args.mmf_epochs,
        mmf_lr=args.mmf_lr,
        mask_mode=args.mask_mode,
        svdvals_method=args.svdvals_method,
        svdvals_k_mult=args.svdvals_k_mult,
        svd_device=args.svd_device,
        use_log10=args.log10,
    )

    print("Done.")


if __name__ == "__main__":
    main()
