"""
vis_identifiability.py

Identifiability / factor-stability visualization via two independent training runs.

The script trains:
  - Standard MF (rotational symmetry; factors can drift across runs),
  - MMF (structure can reduce symmetries; factors tend to align across runs),

and plots an absolute cosine-similarity matrix between the learned left factors (A)
from two runs. Optionally, it reorders columns using the Hungarian algorithm to
correct for permutation ambiguity in the visualization.

Example:
  python vis_identifiability.py --device cuda --out_dir out_iden --I 512 --J 512 --R 40 --K 4 --epochs 2000

Dependencies:
  - torch, numpy, matplotlib
  - scipy (optional, for Hungarian reordering)

Notes:
  - This is a qualitative diagnostic. Stability depends on data, initialization,
    optimization, and hyperparameters.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_fro(X: torch.Tensor) -> torch.Tensor:
    return X / torch.linalg.norm(X).clamp_min(1e-12)


def sample_matrix(I: int, J: int, scale: float, device: str) -> torch.Tensor:
    return scale * torch.randn(I, J, device=device)


def make_block_diag_X(I: int, J: int, num_blocks: int, seed: int, device: str,
                      offblock_noise_scale: float = 0.01) -> torch.Tensor:
    """Block-diagonal matrix with small off-block noise."""
    set_seed(seed)
    X = sample_matrix(I, J, scale=offblock_noise_scale, device=device)

    row_size = I // num_blocks
    col_size = J // num_blocks

    for b in range(num_blocks):
        r0, r1 = b * row_size, (b + 1) * row_size
        c0, c1 = b * col_size, (b + 1) * col_size
        X[r0:r1, c0:c1] = sample_matrix(row_size, col_size, scale=1.0, device=device)

    return normalize_fro(X)


def cosine_similarity_matrix(U1: torch.Tensor, U2: torch.Tensor) -> np.ndarray:
    """Absolute cosine similarity between columns: returns an (R x R) matrix."""
    U1 = U1 / (U1.norm(dim=0, keepdim=True) + 1e-12)
    U2 = U2 / (U2.norm(dim=0, keepdim=True) + 1e-12)
    return (U1.T @ U2).abs().detach().cpu().numpy()


def hungarian_reorder_columns(corr: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Reorder columns of corr via Hungarian algorithm to maximize diagonal mass.

    Returns:
      reordered_corr, col_perm
    """
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
    except Exception:
        return corr, None

    cost = -corr
    row_ind, col_ind = linear_sum_assignment(cost)
    # row_ind is typically [0,1,...,R-1], but we keep the general form.
    reordered = corr[:, col_ind]
    return reordered, col_ind


def plot_heatmap(
    corr: np.ndarray,
    title: str,
    out_path: str,
    x_label: str,
    y_label: str,
    tick_step: int = 8,
) -> None:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(corr, vmin=0.0, vmax=1.0, aspect="equal", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Sparse ticks for readability
    R = corr.shape[0]
    ticks = np.arange(0, R, tick_step) if tick_step > 0 else np.arange(R)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def make_mask_fn(mode: Literal["sigmoid", "gaussian", "cos"]) -> callable:
    if mode == "sigmoid":
        return lambda d: torch.sigmoid(d)
    if mode == "gaussian":
        return lambda d: 4.0 * torch.exp(-d ** 2 / 2)
    if mode == "cos":
        return lambda d: torch.cos(d)
    raise ValueError(f"Unknown mask mode: {mode}")


class BasicMF(nn.Module):
    """Standard matrix factorization: X_hat = A B^T."""
    def __init__(self, I: int, J: int, R: int, device: str):
        super().__init__()
        self.A = nn.Parameter(0.02 * torch.randn(I, R, device=device))
        self.B = nn.Parameter(0.02 * torch.randn(J, R, device=device))

    def forward(self) -> torch.Tensor:
        return self.A @ self.B.T


class MMF(nn.Module):
    """A minimal MMF-style reconstruction model for stability/identifiability visualization."""

    def __init__(
        self,
        I: int,
        J: int,
        R: int,
        K: int,
        device: str,
        mask_mode: Literal["sigmoid", "gaussian", "cos"] = "gaussian",
        fixed_mask_seed: Optional[int] = None,
        learn_shifts: bool = False,
    ):
        super().__init__()
        self.I, self.J, self.R, self.K = I, J, R, K
        self.mask_scale_square = (1.0 / K) ** 2

        self.A = nn.Parameter(0.02 * torch.randn(I, R, device=device).unsqueeze(1))  # (I,1,R)
        self.B = nn.Parameter(0.02 * torch.randn(J, R, device=device).unsqueeze(1))  # (J,1,R)

        # Shifts uA/uB: either fixed (deterministic) or random init and (optionally) learnable.
        if fixed_mask_seed is not None:
            g = torch.Generator(device=device)
            g.manual_seed(int(fixed_mask_seed))
            uA = torch.rand(I, K, generator=g, device=device).unsqueeze(-1) * 2.0
            uB = torch.rand(J, K, generator=g, device=device).unsqueeze(-1) * 2.0
            self.uA = nn.Parameter(uA, requires_grad=False)
            self.uB = nn.Parameter(uB, requires_grad=False)
        else:
            uA = torch.rand(I, K, device=device).unsqueeze(-1) * 2.0
            uB = torch.rand(J, K, device=device).unsqueeze(-1) * 2.0
            self.uA = nn.Parameter(uA, requires_grad=learn_shifts)
            self.uB = nn.Parameter(uB, requires_grad=learn_shifts)

        pos = torch.arange(R, device=device).view(1, 1, R)
        omega = torch.linspace(1.0 / K, 1.0, steps=K, device=device).view(1, K, 1)
        self.register_buffer("template", pos * omega, persistent=False)

        self._mask_fn = make_mask_fn(mask_mode)

    def forward(self) -> torch.Tensor:
        half = float(self.R) / 2.0
        deltaA = self.template - half * self.uA
        deltaB = self.template - half * self.uB
        mA = self._mask_fn(deltaA)
        mB = self._mask_fn(deltaB)

        A_cat = (self.A * mA).reshape(self.I, self.K * self.R)
        B_cat = (self.B * mB).reshape(self.J, self.K * self.R)
        return self.mask_scale_square * (A_cat @ B_cat.T)

    @torch.no_grad()
    def base_left_factor(self) -> torch.Tensor:
        """Return the base left factor (A) of shape (I, R)."""
        return self.A.squeeze(1).detach()


def train_reconstruction(model: nn.Module, X: torch.Tensor, epochs: int, lr: float) -> None:
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        X_hat = model()
        loss = torch.mean((X_hat - X) ** 2)
        loss.backward()
        opt.step()


@dataclass
class RunConfig:
    I: int
    J: int
    R: int
    K: int
    epochs: int
    lr: float
    num_blocks: int
    data_seed: int
    seed_a: int
    seed_b: int
    fixed_mask_seed: Optional[int]
    mask_mode: Literal["sigmoid", "gaussian", "cos"]
    learn_shifts: bool
    reorder: bool
    tick_step: int


def run(cfg: RunConfig, device: str, out_dir: str) -> None:
    ensure_dir(out_dir)

    # Generate a fixed target for both runs.
    X = make_block_diag_X(cfg.I, cfg.J, num_blocks=cfg.num_blocks, seed=cfg.data_seed, device=device)
    X = cfg.K * X  # scale for stronger signal (matches common visualization choices)

    # --------- Standard MF: run A/B ---------
    set_seed(cfg.seed_a)
    mf_a = BasicMF(cfg.I, cfg.J, cfg.R, device=device).to(device)
    train_reconstruction(mf_a, X, epochs=cfg.epochs, lr=cfg.lr)

    set_seed(cfg.seed_b)
    mf_b = BasicMF(cfg.I, cfg.J, cfg.R, device=device).to(device)
    train_reconstruction(mf_b, X, epochs=cfg.epochs, lr=cfg.lr)

    corr_mf = cosine_similarity_matrix(mf_a.A.detach(), mf_b.A.detach())

    # --------- MMF: run A/B ---------
    set_seed(cfg.seed_a)
    mmf_a = MMF(cfg.I, cfg.J, cfg.R, cfg.K, device=device,
                mask_mode=cfg.mask_mode, fixed_mask_seed=cfg.fixed_mask_seed,
                learn_shifts=cfg.learn_shifts).to(device)
    train_reconstruction(mmf_a, X, epochs=cfg.epochs, lr=cfg.lr)

    set_seed(cfg.seed_b)
    mmf_b = MMF(cfg.I, cfg.J, cfg.R, cfg.K, device=device,
                mask_mode=cfg.mask_mode, fixed_mask_seed=cfg.fixed_mask_seed,
                learn_shifts=cfg.learn_shifts).to(device)
    train_reconstruction(mmf_b, X, epochs=cfg.epochs, lr=cfg.lr)

    corr_mmf = cosine_similarity_matrix(mmf_a.base_left_factor(), mmf_b.base_left_factor())

    # Optional reordering for visualization.
    if cfg.reorder:
        corr_mf_plot, _ = hungarian_reorder_columns(corr_mf)
        corr_mmf_plot, _ = hungarian_reorder_columns(corr_mmf)
    else:
        corr_mf_plot, corr_mmf_plot = corr_mf, corr_mmf

    suffix = "reordered" if cfg.reorder else "raw"

    out_mf = os.path.join(out_dir, f"mf_corr_{suffix}.pdf")
    plot_heatmap(
        corr_mf_plot,
        title="Standard MF: cross-run factor similarity",
        out_path=out_mf,
        x_label="latent dim (run B)",
        y_label="latent dim (run A)",
        tick_step=cfg.tick_step,
    )

    out_mmf = os.path.join(out_dir, f"mmf_corr_{suffix}.pdf")
    subtitle = "fixed mask" if cfg.fixed_mask_seed is not None else "random mask"
    plot_heatmap(
        corr_mmf_plot,
        title=f"MMF: cross-run factor similarity ({subtitle})",
        out_path=out_mmf,
        x_label="latent dim (run B)",
        y_label="latent dim (run A)",
        tick_step=cfg.tick_step,
    )

    print(f"[Saved] {out_mf}")
    print(f"[Saved] {out_mmf}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", type=str, default="out_iden")

    p.add_argument("--mode", type=str, default="fixed_mask", choices=["fixed_mask", "random_mask"],
                   help="fixed_mask uses the same uA/uB across runs (via --fixed_mask_seed).")

    # Sizes / training
    p.add_argument("--I", type=int, default=512)
    p.add_argument("--J", type=int, default=512)
    p.add_argument("--R", type=int, default=40)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--lr", type=float, default=0.02)

    # Data
    p.add_argument("--num_blocks", type=int, default=6)
    p.add_argument("--data_seed", type=int, default=999)

    # Two independent training runs
    p.add_argument("--seed_a", type=int, default=101)
    p.add_argument("--seed_b", type=int, default=202)

    # MMF options
    p.add_argument("--mask_mode", type=str, default="gaussian", choices=["sigmoid", "gaussian", "cos"])
    p.add_argument("--learn_shifts", action="store_true", help="Make shifts learnable (slower; changes the test).")
    p.add_argument("--fixed_mask_seed", type=int, default=777,
                   help="Seed for deterministic uA/uB when --mode fixed_mask.")

    # Plotting
    p.add_argument("--reorder", action="store_true",
                   help="Reorder columns using Hungarian matching (requires scipy).")
    p.add_argument("--tick_step", type=int, default=8)
    args = p.parse_args()

    fixed_seed = args.fixed_mask_seed if args.mode == "fixed_mask" else None

    cfg = RunConfig(
        I=args.I,
        J=args.J,
        R=args.R,
        K=args.K,
        epochs=args.epochs,
        lr=args.lr,
        num_blocks=args.num_blocks,
        data_seed=args.data_seed,
        seed_a=args.seed_a,
        seed_b=args.seed_b,
        fixed_mask_seed=fixed_seed,
        mask_mode=args.mask_mode,
        learn_shifts=bool(args.learn_shifts),
        reorder=bool(args.reorder),
        tick_step=args.tick_step,
    )

    run(cfg, device=args.device, out_dir=args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
