"""
main_reconstruction.py

Reconstruction experiments for Masked Mixture Factorization (MMF).

This script benchmarks MMF against common reconstruction baselines on a fully
observed matrix X by minimizing mean squared error (MSE), and reports:
- MSE
- relative Frobenius error ||X_hat - X||_F / ||X||_F
- wall-clock training time

It supports synthetic data generation (random or block-diagonal), parameter-budget
matched comparisons, CSV logging, and optional Matplotlib plots.

Dependencies:
  - Python 3.9+
  - PyTorch
  - Matplotlib (optional, only if plotting is enabled)

Example commands:
  # 1) Budget sweep (vary R, fixed K_ratio)
  python main_reconstruction.py --device cuda --I 256 --J 256 \
      --R_list 16,32,48,64,80 --K_ratio 0.75 --out_dir out_budget

  # 2) K sweep (fixed R, vary K_ratio_list)
  python main_reconstruction.py --device cuda --I 256 --J 256 \
      --R 80 --K_ratio_list 0.1,0.25,0.5,0.75 --out_dir out_ksweep

  # 3) Grid sweep (R_list x K_ratio_list) + plots
  python main_reconstruction.py --device cuda --I 256 --J 256 \
      --R_list 16,32,48,64,80 --K_ratio_list 0.25,0.5,0.75 --out_dir out_grid

  # 4) Block-diagonal data
  python main_reconstruction.py --device cuda --data_mode blockdiag --num_blocks 6 \
      --offblock_noise_scale 0.01 --R_list 80,160,240 --K_ratio 0.4 --out_dir out_block
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Reproducibility utilities
# -----------------------------------------------------------------------------
def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Set seeds for PyTorch (CPU/CUDA). Optionally enable deterministic ops."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Note: enabling deterministic algorithms may raise errors for some ops,
        # and can reduce performance
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


def cuda_sync_if_needed(device: str) -> None:
    """Synchronize CUDA to get accurate timing when using GPU."""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


# -----------------------------------------------------------------------------
# Parsing helpers
# -----------------------------------------------------------------------------
def parse_int_list(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    s = (s or "").strip()
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_str_list(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def relative_fro_error(X_hat: torch.Tensor, X: torch.Tensor) -> float:
    num = torch.linalg.norm(X_hat - X)
    den = torch.linalg.norm(X).clamp_min(1e-12)
    return float((num / den).item())


# -----------------------------------------------------------------------------
# Synthetic data generation
# -----------------------------------------------------------------------------
def _sample_matrix(
    I: int,
    J: int,
    dist: Literal["normal", "uniform"],
    scale: float,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if dist == "normal":
        return scale * torch.randn(I, J, device=device, dtype=dtype)
    if dist == "uniform":
        return scale * (2.0 * torch.rand(I, J, device=device, dtype=dtype) - 1.0)
    raise ValueError(f"Unknown dist: {dist}")


def _normalize_fro(X: torch.Tensor) -> torch.Tensor:
    fro = torch.linalg.norm(X).clamp_min(1e-12)
    return X / fro


def make_random_X(
    I: int,
    J: int,
    dist: Literal["normal", "uniform"] = "normal",
    scale: float = 1.0,
    normalize: Optional[Literal["fro"]] = "fro",
    seed: int = 0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    set_global_seed(seed)
    X = _sample_matrix(I, J, dist=dist, scale=scale, device=device, dtype=dtype)
    if normalize == "fro":
        X = _normalize_fro(X)
    elif normalize is None:
        pass
    else:
        raise ValueError(f"Unknown normalize: {normalize}")
    return X


def _split_sizes(total: int, num_blocks: int) -> List[int]:
    if num_blocks <= 0:
        raise ValueError("num_blocks must be >= 1")
    base = total // num_blocks
    rem = total % num_blocks
    return [base + (1 if b < rem else 0) for b in range(num_blocks)]


def make_block_diag_X(
    I: int,
    J: int,
    num_blocks: int,
    dist: Literal["normal", "uniform"] = "normal",
    block_scale: float = 1.0,
    offblock_noise_scale: float = 0.01,
    normalize: Optional[Literal["fro"]] = "fro",
    seed: int = 0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Construct a (roughly) block-diagonal matrix. Off-block entries are small noise.
    """
    set_global_seed(seed)
    X = _sample_matrix(I, J, dist=dist, scale=offblock_noise_scale, device=device, dtype=dtype)

    row_sizes = _split_sizes(I, num_blocks)
    col_sizes = _split_sizes(J, num_blocks)

    r0, c0 = 0, 0
    for b in range(num_blocks):
        r1 = r0 + row_sizes[b]
        c1 = c0 + col_sizes[b]
        block = _sample_matrix(
            row_sizes[b], col_sizes[b], dist=dist, scale=block_scale, device=device, dtype=dtype
        )
        X[r0:r1, c0:c1] = block
        r0, c0 = r1, c1

    if normalize == "fro":
        X = _normalize_fro(X)
    elif normalize is None:
        pass
    else:
        raise ValueError(f"Unknown normalize: {normalize}")
    return X


# -----------------------------------------------------------------------------
# Baseline: truncated SVD reconstruction
# -----------------------------------------------------------------------------
@torch.no_grad()
def truncated_svd_reconstruct(X: torch.Tensor, r: int) -> torch.Tensor:
    """
    Best rank-r approximation under Frobenius norm (Eckart–Young theorem).
    """
    r = int(max(1, min(r, min(X.shape))))
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    return (U[:, :r] * S[:r]) @ Vh[:r, :]


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class BasicMF(nn.Module):
    """Matrix factorization: X ≈ A B^T"""

    def __init__(self, I: int, J: int, R: int, init_scale: float = 0.02, device: str = "cpu"):
        super().__init__()
        self.A = nn.Parameter(init_scale * torch.randn(I, R, device=device))
        self.B = nn.Parameter(init_scale * torch.randn(J, R, device=device))

    def forward(self) -> torch.Tensor:
        return self.A @ self.B.T

    @torch.no_grad()
    def snapshot(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.A.detach().clone(), self.B.detach().clone())

    @torch.no_grad()
    def load_snapshot(self, snap: Tuple[torch.Tensor, torch.Tensor]) -> None:
        A, B = snap
        self.A.copy_(A)
        self.B.copy_(B)


class BiasMF(nn.Module):
    """Biased MF: X ≈ A B^T + a 1^T + 1 b^T + mu"""

    def __init__(self, I: int, J: int, R: int, init_scale: float = 0.02, device: str = "cpu"):
        super().__init__()
        self.A = nn.Parameter(init_scale * torch.randn(I, R, device=device))
        self.B = nn.Parameter(init_scale * torch.randn(J, R, device=device))
        self.a = nn.Parameter(torch.zeros(I, device=device))
        self.b = nn.Parameter(torch.zeros(J, device=device))
        self.mu = nn.Parameter(torch.zeros((), device=device))

    def forward(self) -> torch.Tensor:
        return self.A @ self.B.T + self.a[:, None] + self.b[None, :] + self.mu

    @torch.no_grad()
    def snapshot(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.A.detach().clone(),
            self.B.detach().clone(),
            self.a.detach().clone(),
            self.b.detach().clone(),
            self.mu.detach().clone(),
        )

    @torch.no_grad()
    def load_snapshot(self, snap) -> None:
        A, B, a, b, mu = snap
        self.A.copy_(A)
        self.B.copy_(B)
        self.a.copy_(a)
        self.b.copy_(b)
        self.mu.copy_(mu)


class NMFModel(nn.Module):
    """
    Nonnegative MF via softplus reparameterization:
      X ≈ A B^T with A,B >= 0.
    """

    def __init__(self, I: int, J: int, R: int, init_scale: float = 0.02, device: str = "cpu"):
        super().__init__()
        self.A_raw = nn.Parameter(init_scale * torch.randn(I, R, device=device))
        self.B_raw = nn.Parameter(init_scale * torch.randn(J, R, device=device))
        self.softplus = nn.Softplus(beta=1.0)

    def forward(self) -> torch.Tensor:
        A = self.softplus(self.A_raw)
        B = self.softplus(self.B_raw)
        return A @ B.T

    @torch.no_grad()
    def snapshot(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.A_raw.detach().clone(), self.B_raw.detach().clone())

    @torch.no_grad()
    def load_snapshot(self, snap: Tuple[torch.Tensor, torch.Tensor]) -> None:
        A_raw, B_raw = snap
        self.A_raw.copy_(A_raw)
        self.B_raw.copy_(B_raw)


def make_mask_fn(mode: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Return a pointwise mask function g(d). Different choices produce different
    gating profiles; some are bounded in [0,1], others are not.
    """
    if mode == "sin":
        return torch.sin
    if mode == "cos":
        return torch.cos
    if mode == "gaussian":
        return lambda d: 4 * torch.exp(-d ** 2 / 2.0)
    if mode == "sigmoid":
        return torch.sigmoid
    if mode == "tanh":
        return torch.tanh
    if mode == "triangle":
        return lambda d: 4 * torch.clamp(1.0 - torch.abs(d), min=0.0)
    if mode == "linear":
        return lambda d: d
    if mode == "sinc":
        return lambda d: 4.0 * torch.sinc(d / math.pi)
    if mode == "square":
        return lambda d: torch.tanh(torch.sin(d) / 0.1) + 0.5
    if mode == "cauchy":
        return lambda d: 1.0 / (1.0 + (4.0 * d) ** 2)
    if mode == "sin01":
        return lambda d: (torch.sin(d) + 1.0) / 2.0
    if mode == "cos01":
        return lambda d: (torch.cos(d) + 1.0) / 2.0
    if mode == "linear01":
        return lambda d: (d + 0.5).clamp(0.0, 1.0)
    if mode == "psin":
        return lambda d: (torch.sin(math.pi * d) + 2.5) / 5.0
    if mode == "pcos":
        return lambda d: (torch.cos(math.pi * d) + 2.5) / 5.0
    if mode == "tri":
        return lambda d: 1.0 - 2.0 * torch.abs(torch.remainder(d + 1.0, 2.0) - 1.0)
    raise ValueError(
        f"Unknown mask mode: {mode}. "
        "Try one of: sin, cos, gaussian, sigmoid, tanh, triangle, linear, sinc, "
        "square, cauchy, sin01, cos01, linear01, psin, pcos, tri."
    )


class MMF(nn.Module):
    """
    Masked Mixture Factorization (MMF).

    We maintain base factors A,B of size R_base and learn K shift parameters per row/column
    to generate K masks. The effective concatenated latent size is K*R_base.

    For budget matching with a vanilla MF of rank R:
      choose R_base = R - K
    because MMF parameters scale as:
      (I+J)*R_base  +  (I+J)*K  ≈  (I+J)*R
    """

    def __init__(
        self,
        I: int,
        J: int,
        R_base: int,
        K: int,
        init_scale: float = 0.02,
        device: str = "cpu",
        mask_mode_A: str = "cos",
        mask_mode_B: str = "cos",
        mask_scale: Optional[float] = None,
    ):
        super().__init__()
        if K < 1:
            raise ValueError("K must be >= 1")
        if R_base < 1:
            raise ValueError("R_base must be >= 1")

        self.I, self.J, self.R_base, self.K = int(I), int(J), int(R_base), int(K)

        # Empirically, scaling by 1/K stabilizes the magnitude of the summed mixtures
        self.mask_scale = float(1.0 / K) if (mask_scale is None) else float(mask_scale)

        # Parameters:
        #   A:  (I, 1, R_base), B: (J, 1, R_base)
        #   uA: (I, K, 1),     uB: (J, K, 1)    (learned shifts)
        self.A = nn.Parameter(init_scale * torch.randn(I, R_base, device=device).unsqueeze(1))
        self.B = nn.Parameter(init_scale * torch.randn(J, R_base, device=device).unsqueeze(1))
        self.uA = nn.Parameter(torch.zeros(I, K, device=device).unsqueeze(-1))
        self.uB = nn.Parameter(torch.zeros(J, K, device=device).unsqueeze(-1))

        # Register constant buffers so they move correctly with .to(device)
        pos = torch.arange(R_base, device=device).view(1, 1, R_base)                # (1,1,R_base)
        omega = torch.linspace(1.0 / K, 1.0, steps=K, device=device).view(1, K, 1)  # (1,K,1)
        template = pos * omega                                                      # (1,K,R_base)
        self.register_buffer("pos", pos, persistent=False)
        self.register_buffer("omega", omega, persistent=False)
        self.register_buffer("template", template, persistent=False)

        self._half = float(R_base) / 2.0
        self._mask_fn_A = make_mask_fn(mask_mode_A)
        self._mask_fn_B = make_mask_fn(mask_mode_B)

    def forward(self) -> torch.Tensor:
        deltaA = self.template - self._half * self.uA
        deltaB = self.template - self._half * self.uB

        mA = self._mask_fn_A(deltaA)
        mB = self._mask_fn_B(deltaB)

        # Concatenate K masked copies along the latent dimension
        A_cat = (self.A * mA).reshape(self.I, self.K * self.R_base)
        B_cat = (self.B * mB).reshape(self.J, self.K * self.R_base)

        # Equivalent to (mask_scale * A_cat) @ (mask_scale * B_cat)^T
        return (self.mask_scale**2) * (A_cat @ B_cat.T)

    @torch.no_grad()
    def snapshot(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.A.detach().clone(),
            self.B.detach().clone(),
            self.uA.detach().clone(),
            self.uB.detach().clone(),
        )

    @torch.no_grad()
    def load_snapshot(self, snap) -> None:
        A, B, uA, uB = snap
        self.A.copy_(A)
        self.B.copy_(B)
        self.uA.copy_(uA)
        self.uB.copy_(uB)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainResult:
    best_epoch: int
    train_time_sec: float
    mse: float
    rel_fro: float


def train_full_matrix(
    model: nn.Module,
    X: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    log_every: int,
    seed: int,
    device: str,
    ridge_lambda: float = 0.0,
    ridge_params: Optional[Sequence[torch.Tensor]] = None,
    tag: str = "",
) -> TrainResult:
    """
    Train a model to reconstruct a fully observed matrix X by minimizing MSE.
    Optionally adds ridge regularization on specified parameters.
    """
    set_global_seed(seed)
    n_elem = float(X.numel())
    x_fro = float(torch.linalg.norm(X).detach().item())

    best_val = float("inf")
    best_epoch = -1
    best_snap = None

    cuda_sync_if_needed(device)
    t_start = time.time()
    t_last = t_start

    for epoch in range(int(epochs)):
        X_hat = model()
        diff = X_hat - X
        mse = (diff * diff).mean()

        loss = mse
        if ridge_lambda > 0.0 and ridge_params:
            ridge = torch.zeros((), device=X.device, dtype=X.dtype)
            for p in ridge_params:
                ridge = ridge + (p * p).sum()
            loss = loss + float(ridge_lambda) * ridge / max(n_elem, 1.0)

        val = float(loss.detach().item())
        if val < best_val:
            best_val = val
            best_epoch = epoch
            if hasattr(model, "snapshot"):
                best_snap = model.snapshot()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if log_every > 0 and ((epoch % log_every == 0) or (epoch == epochs - 1)):
            rel_fro_approx = math.sqrt(float(mse.detach().item()) * n_elem) / max(x_fro, 1e-12)
            best_rel_fro_approx = math.sqrt(best_val * n_elem) / max(x_fro, 1e-12)
            dt = time.time() - t_last
            print(
                f"[{tag:12s} {epoch:5d}/{epochs}] "
                f"loss={loss.item():.6e}  rel_fro~={rel_fro_approx:.6f}  "
                f"(best~={best_rel_fro_approx:.6f}@{best_epoch})  step_time={dt:.3f}s"
            )
            t_last = time.time()

    cuda_sync_if_needed(device)
    train_time = time.time() - t_start

    if best_snap is not None and hasattr(model, "load_snapshot"):
        model.load_snapshot(best_snap)

    with torch.no_grad():
        X_hat = model()
        mse_final = float(torch.mean((X_hat - X) ** 2).item())
        rel_final = relative_fro_error(X_hat, X)

    return TrainResult(best_epoch=best_epoch, train_time_sec=train_time, mse=mse_final, rel_fro=rel_final)


# -----------------------------------------------------------------------------
# Method wrappers
# -----------------------------------------------------------------------------
def train_mf_sgd(
    X: torch.Tensor,
    I: int,
    J: int,
    R: int,
    device: str,
    epochs: int,
    lr: float,
    log_every: int,
    seed: int,
    momentum: float = 0.9,
) -> Tuple[nn.Module, TrainResult]:
    model = BasicMF(I, J, R, device=device).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    res = train_full_matrix(model, X, opt, epochs, log_every, seed, device=device, tag="MF-SGD")
    return model, res


def train_mf_ridge(
    X: torch.Tensor,
    I: int,
    J: int,
    R: int,
    device: str,
    epochs: int,
    lr: float,
    log_every: int,
    seed: int,
    ridge_lambda: float = 1e-3,
) -> Tuple[nn.Module, TrainResult]:
    model = BasicMF(I, J, R, device=device).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    res = train_full_matrix(
        model,
        X,
        opt,
        epochs,
        log_every,
        seed,
        device=device,
        ridge_lambda=ridge_lambda,
        ridge_params=[model.A, model.B],
        tag="MF-Ridge",
    )
    return model, res


def train_mf_bias(
    X: torch.Tensor,
    I: int,
    J: int,
    R: int,
    device: str,
    epochs: int,
    lr: float,
    log_every: int,
    seed: int,
    bias_ridge_lambda: float = 0.0,
) -> Tuple[nn.Module, TrainResult]:
    model = BiasMF(I, J, R, device=device).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ridge_params = [model.A, model.B, model.a, model.b]
    res = train_full_matrix(
        model,
        X,
        opt,
        epochs,
        log_every,
        seed,
        device=device,
        ridge_lambda=bias_ridge_lambda,
        ridge_params=ridge_params,
        tag="MF+Bias",
    )
    return model, res


def train_nmf(
    X: torch.Tensor,
    I: int,
    J: int,
    R: int,
    device: str,
    epochs: int,
    lr: float,
    log_every: int,
    seed: int,
) -> Tuple[nn.Module, TrainResult]:
    model = NMFModel(I, J, R, device=device).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    res = train_full_matrix(model, X, opt, epochs, log_every, seed, device=device, tag="NMF")
    return model, res


def train_mmf(
    X: torch.Tensor,
    I: int,
    J: int,
    R_base: int,
    K: int,
    device: str,
    epochs: int,
    lr_factors: float,
    lr_shifts: float,
    log_every: int,
    seed: int,
    mask_mode_A: str,
    mask_mode_B: str,
    mask_scale: Optional[float],
) -> Tuple[nn.Module, TrainResult]:
    model = MMF(
        I=I,
        J=J,
        R_base=R_base,
        K=K,
        device=device,
        mask_mode_A=mask_mode_A,
        mask_mode_B=mask_mode_B,
        mask_scale=mask_scale,
    ).to(device)

    opt = torch.optim.Adam(
        [
            {"params": [model.A, model.B], "lr": lr_factors},
            {"params": [model.uA, model.uB], "lr": lr_shifts},
        ]
    )
    res = train_full_matrix(model, X, opt, epochs, log_every, seed, device=device, tag=f"MMF(K={K})")
    return model, res


# -----------------------------------------------------------------------------
# Budget mapping
# -----------------------------------------------------------------------------
def budget_from_R(I: int, J: int, R: int) -> int:
    """Parameter budget proxy for vanilla MF: (I+J)*R."""
    return int((I + J) * R)


def rank_for_bias_mf(I: int, J: int, budget: int, rank_bonus: int = 0) -> int:
    """
    Compute the MF rank that approximately matches the parameter budget of vanilla MF
    when adding bias terms (a, b, mu).
    """
    extra = I + J + 1
    denom = I + J
    r = (budget - extra) // denom
    return int(max(1, r + int(rank_bonus)))


# -----------------------------------------------------------------------------
# One experiment setting
# -----------------------------------------------------------------------------
def run_one_setting(
    X_orig: torch.Tensor,
    I: int,
    J: int,
    R: int,
    K: int,
    args: argparse.Namespace,
    methods: Sequence[str],
    seed: int,
) -> List[Dict[str, object]]:
    """
    Run all selected methods under a single (R, K, seed) configuration.

    Notes on scaling:
      For some mask families and large K, the magnitude of X can affect optimization.
      We optionally rescale X to have a target Frobenius norm in the scaled space.
      Results are always reported back on the original scale.
    """
    device = args.device
    budget = budget_from_R(I, J, R)

    target_fro = float(K) if args.x_target_fro < 0 else float(args.x_target_fro)

    X = X_orig
    x_fro = float(torch.linalg.norm(X).detach().item())
    scale_X = target_fro / max(x_fro, 1e-12)
    X_scaled = X * scale_X

    print("\n" + "=" * 72)
    print(f"[Setting] seed={seed}  I={I}  J={J}  R={R}  K={K}  budget~={(budget):,}")
    print(f"Scaling: target_fro={target_fro:.3f}  scale_X={scale_X:.6f}")
    print("=" * 72)

    rows: List[Dict[str, object]] = []

    def add_row(method: str, params: int, result: TrainResult, Xhat_scaled: torch.Tensor) -> None:
        Xhat = Xhat_scaled / scale_X  # back to original scale
        mse = float(torch.mean((Xhat - X) ** 2).item())
        rel = relative_fro_error(Xhat, X)
        rows.append(
            {
                "seed": int(seed),
                "I": int(I),
                "J": int(J),
                "R": int(R),
                "K": int(K),
                "K_ratio": float(K / R),
                "budget": int(budget),
                "method": str(method),
                "params": int(params),
                "best_epoch": int(result.best_epoch),
                "train_time_sec": float(result.train_time_sec),
                "mse": float(mse),
                "rel_fro": float(rel),
            }
        )

    # -------------------------------------------------------------------------
    # MMF (budget matched by choosing R_base = R - K)
    # -------------------------------------------------------------------------
    if "mmf" in methods:
        R_base = R - K
        if R_base < 1:
            raise ValueError(f"Invalid configuration: R - K must be >= 1 (R={R}, K={K}).")

        mask_scale = None if (args.mask_scale < 0) else float(args.mask_scale)

        model, res = train_mmf(
            X=X_scaled,
            I=I,
            J=J,
            R_base=R_base,
            K=K,
            device=device,
            epochs=args.epochs,
            lr_factors=args.lr_factors,
            lr_shifts=args.lr_shifts,
            log_every=args.log_every,
            seed=seed,
            mask_mode_A=args.mask_mode_A,
            mask_mode_B=args.mask_mode_B,
            mask_scale=mask_scale,
        )
        with torch.no_grad():
            Xhat_s = model()
        add_row("MMF", count_parameters(model), res, Xhat_s)

    # -------------------------------------------------------------------------
    # Truncated SVD
    # -------------------------------------------------------------------------
    if "svd" in methods:
        cuda_sync_if_needed(device)
        t0 = time.time()
        Xhat_s = truncated_svd_reconstruct(X_scaled, R)
        cuda_sync_if_needed(device)
        tsec = time.time() - t0

        # Proxy parameter count for consistency in tables (ignoring singular values)
        params = (I + J) * R
        print(f"[SVD rank={R}] rel_fro_scaled={relative_fro_error(Xhat_s, X_scaled):.6f}  time={tsec:.3f}s")
        add_row("SVD", params, TrainResult(best_epoch=0, train_time_sec=tsec, mse=0.0, rel_fro=0.0), Xhat_s)

    # -------------------------------------------------------------------------
    # MF baselines
    # -------------------------------------------------------------------------
    if "mf_sgd" in methods:
        model, res = train_mf_sgd(
            X=X_scaled,
            I=I,
            J=J,
            R=R,
            device=device,
            epochs=args.epochs,
            lr=args.lr_mf_sgd,
            log_every=args.log_every,
            seed=seed,
            momentum=args.mf_sgd_momentum,
        )
        with torch.no_grad():
            Xhat_s = model()
        add_row("MF_SGD", count_parameters(model), res, Xhat_s)

    if "mf_ridge" in methods:
        model, res = train_mf_ridge(
            X=X_scaled,
            I=I,
            J=J,
            R=R,
            device=device,
            epochs=args.epochs,
            lr=args.lr_mf_ridge,
            log_every=args.log_every,
            seed=seed,
            ridge_lambda=args.ridge_lambda,
        )
        with torch.no_grad():
            Xhat_s = model()
        add_row("MF_Ridge", count_parameters(model), res, Xhat_s)

    if "mf_bias" in methods:
        r_bias = rank_for_bias_mf(I, J, budget, rank_bonus=args.bias_rank_bonus)
        print(f"[MF+Bias] budget match: R_base={R} -> R_bias={r_bias} (bonus={args.bias_rank_bonus})")
        model, res = train_mf_bias(
            X=X_scaled,
            I=I,
            J=J,
            R=r_bias,
            device=device,
            epochs=args.epochs,
            lr=args.lr_mf_bias,
            log_every=args.log_every,
            seed=seed,
            bias_ridge_lambda=args.bias_ridge_lambda,
        )
        with torch.no_grad():
            Xhat_s = model()
        add_row("MF_Bias", count_parameters(model), res, Xhat_s)

    # -------------------------------------------------------------------------
    # NMF (only meaningful when X is nonnegative)
    # -------------------------------------------------------------------------
    if "nmf" in methods:
        if float(X_scaled.min().item()) < -1e-12:
            print("[NMF] Skipped: X has negative entries. Use --data_nonneg for a fair NMF run.")
        else:
            model, res = train_nmf(
                X=X_scaled,
                I=I,
                J=J,
                R=R,
                device=device,
                epochs=args.epochs,
                lr=args.lr_nmf,
                log_every=args.log_every,
                seed=seed,
            )
            with torch.no_grad():
                Xhat_s = model()
            add_row("NMF", count_parameters(model), res, Xhat_s)

    return rows


# -----------------------------------------------------------------------------
# Output utilities
# -----------------------------------------------------------------------------
def save_csv(rows: List[Dict[str, object]], path: str) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# -----------------------------------------------------------------------------
# Plotting (optional)
# -----------------------------------------------------------------------------
def make_plots(rows: List[Dict[str, object]], out_dir: str, backend: str = "Agg") -> None:
    """
    Creates:
      (1) rel_fro vs budget for each fixed K_ratio (mean over seeds)
      (2) rel_fro vs K_ratio for each fixed R (MMF curve + baseline horizontal lines)
    """
    if not rows:
        return

    import matplotlib

    # Use a non-interactive backend by default so plotting works on servers/CI
    matplotlib.use(backend)
    import matplotlib.pyplot as plt
    from collections import defaultdict

    marker_map = {
        "MMF": "o",
        "SVD": "s",
        "MF_SGD": "^",
        "MF_Ridge": "D",
        "MF_Bias": "v",
        "NMF": "P",
    }
    fallback_markers = ["X", "*", "<", ">", "h", "H", "8", "p", "+", "x", "1", "2", "3", "4"]

    def label_of(method: str) -> str:
        return "MMF (proposed)" if method == "MMF" else method

    def order_methods(method_list: Sequence[str]) -> List[str]:
        rest = sorted([m for m in method_list if m != "MMF"])
        return (["MMF"] if "MMF" in method_list else []) + rest

    def marker_of(method: str, idx_fallback: int) -> str:
        return marker_map.get(method, fallback_markers[idx_fallback % len(fallback_markers)])

    # Aggregate mean over seeds for each (R, K, method)
    acc: Dict[Tuple[int, int, str], List[float]] = defaultdict(list)
    meta: Dict[Tuple[int, int, str], Dict[str, object]] = {}
    for r in rows:
        key = (int(r["R"]), int(r["K"]), str(r["method"]))
        acc[key].append(float(r["rel_fro"]))
        meta[key] = r

    agg: List[Dict[str, object]] = []
    for (R, K, method), vals in acc.items():
        agg.append(
            {
                "R": R,
                "K": K,
                "K_ratio": K / R,
                "budget": int(meta[(R, K, method)]["budget"]),
                "method": method,
                "rel_fro_mean": sum(vals) / len(vals),
                "nseed": len(vals),
            }
        )

    methods = order_methods(sorted({a["method"] for a in agg}))

    # (1) rel_fro vs budget for each fixed K_ratio
    grp_kr: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for a in agg:
        grp_kr[f"{float(a['K_ratio']):.4f}"].append(a)

    for kr_key, items in sorted(grp_kr.items(), key=lambda kv: float(kv[0])):
        plt.figure()

        fallback_idx = 0
        for method in methods:
            pts = [x for x in items if x["method"] == method]
            if not pts:
                continue
            pts = sorted(pts, key=lambda d: int(d["budget"]))
            xs = [int(p["budget"]) for p in pts]
            ys = [float(p["rel_fro_mean"]) for p in pts]

            mk = marker_of(method, fallback_idx)
            if method not in marker_map:
                fallback_idx += 1

            plt.plot(
                xs,
                ys,
                marker=mk,
                markerfacecolor="none",
                markeredgewidth=1.5,
                linewidth=1.8,
                label=label_of(method),
            )

        plt.xlabel("Parameter budget ~ (I+J)*B")
        plt.ylabel("Relative Frobenius error")
        plt.title(f"rel_fro vs budget (K_ratio={kr_key})")
        plt.grid(True, alpha=0.3)
        plt.legend()

        fn = os.path.join(out_dir, f"plot_rel_fro_vs_budget_Kratio{kr_key}.png")
        plt.savefig(fn, dpi=200, bbox_inches="tight")
        plt.close()

    # (2) rel_fro vs K_ratio for each fixed R
    grp_R: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for a in agg:
        grp_R[int(a["R"])].append(a)

    for R, items in sorted(grp_R.items(), key=lambda kv: kv[0]):
        plt.figure()

        # MMF curve first (so it appears first in the legend)
        mmf_pts = [x for x in items if x["method"] == "MMF"]
        if mmf_pts:
            mmf_pts = sorted(mmf_pts, key=lambda d: float(d["K_ratio"]))
            xs = [float(p["K_ratio"]) for p in mmf_pts]
            ys = [float(p["rel_fro_mean"]) for p in mmf_pts]
            plt.plot(
                xs,
                ys,
                marker=marker_map.get("MMF", "o"),
                markerfacecolor="none",
                markeredgewidth=1.5,
                linewidth=1.8,
                label=label_of("MMF"),
            )

        # Baselines as horizontal lines
        for method in methods:
            if method == "MMF":
                continue
            pts = [x for x in items if x["method"] == method]
            if not pts:
                continue
            y = sum(float(p["rel_fro_mean"]) for p in pts) / len(pts)
            plt.hlines(y, xmin=0.0, xmax=1.0, linestyles="--", linewidth=1.5, label=label_of(method))

        plt.xlabel("K_ratio (=K/R)")
        plt.ylabel("Relative Frobenius error")
        plt.title(f"rel_fro vs K_ratio (R={R})")
        plt.xlim(0.0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend()

        fn = os.path.join(out_dir, f"plot_rel_fro_vs_Kratio_R{R}.png")
        plt.savefig(fn, dpi=200, bbox_inches="tight")
        plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reconstruction experiments for MMF and baselines.")

    # Matrix shape
    p.add_argument("--I", type=int, default=1080, help="Number of rows.")
    p.add_argument("--J", type=int, default=1080, help="Number of columns.")

    # Sweeps
    p.add_argument("--R", type=int, default=80, help="Vanilla MF rank (budget proxy).")
    p.add_argument("--R_list", type=str, default="80,160,240", help="Comma-separated ranks, e.g., 16,32,64.")
    p.add_argument("--K", type=int, default=40, help="Number of masks (used when K_ratio is not provided).")
    p.add_argument("--K_ratio", type=float, default=0.4, help="If >=0, use K=round(K_ratio*R).")
    p.add_argument("--K_ratio_list", type=str, default="", help="Comma-separated K_ratio values, e.g., 0.25,0.5,0.75.")

    # Methods
    p.add_argument(
        "--methods",
        type=str,
        default="mmf,svd,mf_sgd,mf_ridge,mf_bias,nmf",
        help="Comma-separated methods: mmf, svd, mf_sgd, mf_ridge, mf_bias, nmf.",
    )

    # Optimization
    p.add_argument("--epochs", type=int, default=2000, help="Training epochs for gradient-based methods.")
    p.add_argument("--log_every", type=int, default=200, help="Log every N epochs (0 to disable).")

    p.add_argument("--lr_factors", type=float, default=2e-2, help="Learning rate for MMF factors (A,B).")
    p.add_argument("--lr_shifts", type=float, default=1e-2, help="Learning rate for MMF shifts (uA,uB).")

    p.add_argument("--lr_mf_sgd", type=float, default=2e-1, help="Learning rate for MF-SGD baseline.")
    p.add_argument("--mf_sgd_momentum", type=float, default=0.999, help="Momentum for MF-SGD baseline.")

    p.add_argument("--lr_mf_ridge", type=float, default=2e-2, help="Learning rate for MF-Ridge baseline.")
    p.add_argument("--ridge_lambda", type=float, default=1e-3, help="Ridge coefficient for MF-Ridge baseline.")

    p.add_argument("--lr_mf_bias", type=float, default=2e-2, help="Learning rate for MF+Bias baseline.")
    p.add_argument("--bias_ridge_lambda", type=float, default=0.0, help="Optional ridge for MF+Bias baseline.")
    p.add_argument("--bias_rank_bonus", type=int, default=0, help="Optional rank bonus for MF+Bias.")

    p.add_argument("--lr_nmf", type=float, default=2e-2, help="Learning rate for NMF baseline.")

    # Mask configuration
    p.add_argument("--mask_mode_A", type=str, default="gaussian", help="Mask family for rows (A).")
    p.add_argument("--mask_mode_B", type=str, default="gaussian", help="Mask family for columns (B).")
    p.add_argument("--mask_scale", type=float, default=-1.0, help="If >=0, override default mask scale (1/K).")

    # Data
    p.add_argument("--data_mode", type=str, default="blockdiag", choices=["random", "blockdiag"])
    p.add_argument("--dist", type=str, default="normal", choices=["normal", "uniform"])
    p.add_argument("--data_nonneg", action="store_true", help="Make X nonnegative (useful for NMF).")
    p.add_argument("--num_blocks", type=int, default=6, help="Number of blocks for block-diagonal data.")
    p.add_argument("--block_scale", type=float, default=1.0, help="Scale for diagonal blocks.")
    p.add_argument("--offblock_noise_scale", type=float, default=0.01, help="Noise scale for off-block entries.")

    # Scaling
    p.add_argument(
        "--x_target_fro",
        type=float,
        default=-1.0,
        help="If <0, set target Frobenius norm to K. Otherwise use the provided value.",
    )

    # Misc
    p.add_argument("--seed", type=int, default=42, help="Default random seed.")
    p.add_argument("--seeds", type=str, default="", help="Optional comma-separated seeds (overrides --seed).")
    p.add_argument("--deterministic", action="store_true", help="Enable deterministic algorithms in PyTorch.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tf32", action="store_true", help="Allow TF32 matmul on Ampere+ GPUs (faster, slightly less precise).")
    p.add_argument("--out_dir", type=str, default="out", help="Output directory for CSV/plots.")
    p.add_argument("--no_plots", action="store_true", help="Disable plot generation.")
    p.add_argument("--mpl_backend", type=str, default="Agg", help="Matplotlib backend (default: Agg).")

    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    ensure_dir(args.out_dir)

    if args.device.startswith("cuda") and args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Seeds
    seeds = parse_int_list(args.seeds) or [int(args.seed)]

    # R sweep list
    R_list = parse_int_list(args.R_list) or [int(args.R)]

    # K_ratio sweep list
    K_ratio_list = parse_float_list(args.K_ratio_list)
    if not K_ratio_list:
        if args.K_ratio >= 0:
            K_ratio_list = [float(args.K_ratio)]
        else:
            K_ratio_list = [float("nan")]  # sentinel: use fixed K

    # Methods
    methods = [m.lower() for m in parse_str_list(args.methods)]
    valid = {"mmf", "svd", "mf_sgd", "mf_ridge", "mf_bias", "nmf"}
    for m in methods:
        if m not in valid:
            raise ValueError(f"Unknown method '{m}'. Valid methods: {sorted(valid)}")

    # Data creation (shared across runs)
    set_global_seed(seeds[0], deterministic=args.deterministic)
    if args.data_mode == "random":
        X = make_random_X(args.I, args.J, dist=args.dist, normalize="fro", seed=seeds[0], device=args.device)
    else:
        X = make_block_diag_X(
            args.I,
            args.J,
            num_blocks=args.num_blocks,
            dist=args.dist,
            block_scale=args.block_scale,
            offblock_noise_scale=args.offblock_noise_scale,
            normalize="fro",
            seed=seeds[0],
            device=args.device,
        )

    if args.data_nonneg:
        X = X - X.min()
        X = X.clamp_min(0.0)
        X = _normalize_fro(X)

    print(f"Device: {args.device}")
    print(f"Data: mode={args.data_mode}, dist={args.dist}, nonneg={args.data_nonneg}")
    print(f"Methods: {methods}")
    print(f"Output dir: {args.out_dir}")

    all_rows: List[Dict[str, object]] = []

    # Grid: for each (R, K_ratio) and each seed
    for R in R_list:
        for kr in K_ratio_list:
            if math.isnan(kr):
                K = int(args.K)
            else:
                K = int(round(float(kr) * R))
            # K must satisfy 1 <= K <= R-1 for R_base = R-K >= 1
            K = max(1, min(K, int(R) - 1))

            for sd in seeds:
                rows = run_one_setting(
                    X_orig=X,
                    I=int(args.I),
                    J=int(args.J),
                    R=int(R),
                    K=int(K),
                    args=args,
                    methods=methods,
                    seed=int(sd),
                )
                all_rows.extend(rows)

    # Save CSV
    csv_name = f"results_I{args.I}_J{args.J}_{args.data_mode}_mask{args.mask_mode_A}-{args.mask_mode_B}.csv"
    csv_path = os.path.join(args.out_dir, csv_name)
    save_csv(all_rows, csv_path)
    print(f"\nSaved CSV: {csv_path}")

    # Plots
    if not args.no_plots:
        try:
            make_plots(all_rows, args.out_dir, backend=args.mpl_backend)
            print(f"Saved plots to: {args.out_dir}")
        except ModuleNotFoundError:
            print("Matplotlib not found. Skipping plots. Install matplotlib or use --no_plots.")

    print("\nDone.")


if __name__ == "__main__":
    main()
