"""
main_completion.py

Unified MF vs. MMF benchmark for explicit-rating matrix completion.

Supported datasets:
  - MovieLens: ml-100k / ml-1m / ml-10m (from GroupLens)
  - Flixster / Douban: fixed train/test splits from the GC-MC/MGCNN preprocessing (.mat)

Example:
  python main_completion.py --dataset ml-1m --download --R 1200 --K 64 --device cuda

Outputs (saved under --save_dir):
  - Per-seed JSON: <dataset>_R{R}_K{K}_seed{seed}.json
  - Aggregate JSON: <dataset>_R{R}_K{K}_agg.json
  - Optional training curve PNG: <dataset>_curves_R{R}_K{K}_seed{seed}.png (enable with --plot_curves)
"""

import argparse
import math
import json
import os
import random
import time
import zipfile
import urllib.request
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def rmse_mae(pred: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    diff = pred - y
    rmse = torch.sqrt(torch.mean(diff * diff)).item()
    mae = torch.mean(torch.abs(diff)).item()
    return rmse, mae


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def cpu_state_dict_clone(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def download_url(url: str, dst_path: str):
    ensure_dir(os.path.dirname(dst_path) if os.path.dirname(dst_path) else ".")
    if not os.path.isfile(dst_path):
        print(f"[Download] {url}")
        urllib.request.urlretrieve(url, dst_path)
    else:
        print(f"[Download] exists: {dst_path}")


def unzip_if_needed(zip_path: str, extract_to: str, sentinel_dir: Optional[str] = None):
    """Unzip a file if extraction is not already done.

    If ``sentinel_dir`` is provided and exists, we assume the archive has already been extracted.
    Otherwise, if ``extract_to`` is non-empty, we also assume extraction is done.
    """
    if sentinel_dir is not None and os.path.isdir(sentinel_dir):
        return
    if sentinel_dir is None and os.path.isdir(extract_to) and len(os.listdir(extract_to)) > 0:
        return

    ensure_dir(extract_to)
    print(f"[Extract] {zip_path} -> {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def find_files_recursive(root: str, predicate) -> List[str]:
    hits = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            if predicate(p):
                hits.append(p)
    hits.sort()
    return hits


# -----------------------------
# MovieLens download + parse
# -----------------------------
MOVIELENS_URLS = {
    "ml-100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    # MovieLens 10M dataset zip often contains folder "ml-10M100K"
    "ml-10m": "https://files.grouplens.org/datasets/movielens/ml-10m.zip",
}


def _rm_tree(path: str):
    if not os.path.exists(path):
        return
    for dp, dn, fn in os.walk(path, topdown=False):
        for f in fn:
            os.remove(os.path.join(dp, f))
        for d in dn:
            os.rmdir(os.path.join(dp, d))
    os.rmdir(path)


def download_and_extract_movielens(dataset: str, data_dir: str) -> str:
    assert dataset in MOVIELENS_URLS, f"Unsupported MovieLens: {dataset}"
    ensure_dir(data_dir)
    url = MOVIELENS_URLS[dataset]
    zip_path = os.path.join(data_dir, f"{dataset}.zip")
    extract_dir = os.path.join(data_dir, dataset)

    if not os.path.isdir(extract_dir):
        download_url(url, zip_path)

        tmp_extract = os.path.join(data_dir, f"{dataset}")
        if os.path.isdir(tmp_extract):
            _rm_tree(tmp_extract)

        unzip_if_needed(zip_path, tmp_extract, sentinel_dir=None)

        inner_dirs = [os.path.join(tmp_extract, d) for d in os.listdir(tmp_extract)
                      if os.path.isdir(os.path.join(tmp_extract, d))]
        if len(inner_dirs) != 1:
            raise RuntimeError(f"Unexpected zip structure: {zip_path} -> {inner_dirs}")
        inner = inner_dirs[0]

        if os.path.exists(extract_dir):
            _rm_tree(extract_dir)

        os.rename(inner, extract_dir)
        _rm_tree(tmp_extract)

    return extract_dir


def load_movielens_ratings(dataset: str, data_dir: str, download: bool = True):
    if download:
        extract_dir = download_and_extract_movielens(dataset, data_dir)
    else:
        extract_dir = os.path.join(data_dir, dataset)
        if not os.path.isdir(extract_dir):
            raise FileNotFoundError(f"Not found: {extract_dir} (set --download or fix --data_dir)")

    rows = []
    if dataset == "ml-100k":
        path = os.path.join(extract_dir, "u.data")
        with open(path, "r", encoding="latin-1") as f:
            for line in f:
                u, i, r, _ = line.strip().split("\t")
                rows.append((int(u), int(i), float(r)))
    elif dataset in ["ml-1m", "ml-10m"]:
        path = os.path.join(extract_dir, "ratings.dat")
        if not os.path.isfile(path):
            cand = find_files_recursive(extract_dir, lambda p: os.path.basename(p).lower() == "ratings.dat")
            if len(cand) == 0:
                raise FileNotFoundError(f"ratings.dat not found under {extract_dir}")
            path = cand[0]

        with open(path, "r", encoding="latin-1") as f:
            for line in f:
                u, i, r, _ = line.strip().split("::")
                rows.append((int(u), int(i), float(r)))
    else:
        raise ValueError(dataset)

    u_raw = np.array([x[0] for x in rows], dtype=np.int64)
    i_raw = np.array([x[1] for x in rows], dtype=np.int64)
    r = np.array([x[2] for x in rows], dtype=np.float32)

    u_unique = np.unique(u_raw)
    i_unique = np.unique(i_raw)
    u_map = {uid: idx for idx, uid in enumerate(u_unique)}
    i_map = {iid: idx for idx, iid in enumerate(i_unique)}

    u = np.array([u_map[x] for x in u_raw], dtype=np.int64)
    i = np.array([i_map[x] for x in i_raw], dtype=np.int64)

    return u, i, r, len(u_unique), len(i_unique)


def per_user_split(u: np.ndarray, i: np.ndarray, r: np.ndarray,
                   num_users: int,
                   val_ratio: float = 0.1,
                   test_ratio: float = 0.1,
                   seed: int = 0):
    rng = np.random.default_rng(seed)
    user_to_idx = [[] for _ in range(num_users)]
    for idx in range(len(u)):
        user_to_idx[u[idx]].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for user in range(num_users):
        idxs = user_to_idx[user]
        if len(idxs) < 3:
            train_idx.extend(idxs)
            continue
        idxs = np.array(idxs, dtype=np.int64)
        rng.shuffle(idxs)

        n = len(idxs)
        n_test = max(1, int(round(n * test_ratio)))
        n_val = max(1, int(round(n * val_ratio))) if val_ratio > 0 else 0
        if n_test + n_val >= n:
            n_test = 1
            n_val = 1 if val_ratio > 0 else 0

        test_part = idxs[:n_test]
        val_part = idxs[n_test:n_test + n_val] if n_val > 0 else np.array([], dtype=np.int64)
        train_part = idxs[n_test + n_val:]

        test_idx.extend(test_part.tolist())
        val_idx.extend(val_part.tolist())
        train_idx.extend(train_part.tolist())

    def take(arr, idxs_):
        return arr[np.array(idxs_, dtype=np.int64)]

    return (
        take(u, train_idx), take(i, train_idx), take(r, train_idx),
        take(u, val_idx), take(i, val_idx), take(r, val_idx),
        take(u, test_idx), take(i, test_idx), take(r, test_idx),
    )


# -----------------------------
# Flixster/Douban common preprocessing (mgcnn repo)
# -----------------------------
def download_and_extract_mgcnn(data_dir: str) -> str:
    ensure_dir(data_dir)
    url = "https://github.com/fmonti/mgcnn/archive/refs/heads/master.zip"
    zip_path = os.path.join(data_dir, "mgcnn-master.zip")

    if not os.path.isfile(zip_path):
        print(f"[Download] {url}")
        urllib.request.urlretrieve(url, zip_path)

    sentinel = os.path.join(data_dir, "mgcnn-master")
    unzip_if_needed(zip_path, data_dir, sentinel_dir=sentinel)

    # GitHub ZIPs may extract into folders like "mgcnn-main" or "mgcnn-master"; normalize the folder name.
    if os.path.isdir(sentinel):
        return sentinel

    cands = [os.path.join(data_dir, d) for d in os.listdir(data_dir)
             if os.path.isdir(os.path.join(data_dir, d)) and d.lower().startswith("mgcnn-")]
    if not cands:
        raise FileNotFoundError(f"Extraction seems failed: no mgcnn-* folder under {data_dir}")
    cands.sort(key=lambda p: len(p))
    return cands[0]


def load_mat_any(path: str) -> dict:
    """
    Try scipy.io.loadmat first (MAT v5). If MAT v7.3, try mat73 (optional).
    """
    try:
        import scipy.io
        return scipy.io.loadmat(path)
    except NotImplementedError:
        try:
            import mat73
            return mat73.loadmat(path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load MAT file (possibly v7.3): {path}\n"
                f"Try: pip install mat73 h5py\nOriginal error: {e}"
            )


def _as_scipy_sparse(x):
    try:
        import scipy.sparse as sp
        return sp.issparse(x)
    except Exception:
        return False


def _to_csr(x):
    import scipy.sparse as sp
    if sp.isspmatrix_csr(x):
        return x
    if sp.issparse(x):
        return x.tocsr()
    return None


def _pick_matrix_like(data: dict, prefer_names: List[str], min_dim: int = 2):
    for name in prefer_names:
        if name in data:
            return name, data[name]

    best = None
    best_name = None
    best_size = -1

    for k, v in data.items():
        if k.startswith("__"):
            continue
        if isinstance(v, (np.ndarray,)) and v.ndim >= min_dim:
            size = int(np.prod(v.shape[:2]))
            if size > best_size:
                best = v
                best_name = k
                best_size = size
        else:
            if _as_scipy_sparse(v):
                size = int(v.shape[0] * v.shape[1])
                if size > best_size:
                    best = v
                    best_name = k
                    best_size = size

    return best_name, best


def load_flixster_douban_from_mgcnn(dataset: str, data_dir: str, download: bool = True):
    """
    Load Flixster/Douban using commonly used preprocessed .mat from mgcnn repo.
    Returns fixed (train_u, train_i, train_r, test_u, test_i, test_r, num_users, num_items)
    """
    assert dataset in ["flixster", "douban"], "Only flixster/douban here."

    if download:
        root = download_and_extract_mgcnn(data_dir)
    else:
        # allow any mgcnn-* folder
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Not found: {data_dir} (fix --data_dir or set --download)")
        cands = [os.path.join(data_dir, d) for d in os.listdir(data_dir)
                 if os.path.isdir(os.path.join(data_dir, d)) and d.lower().startswith("mgcnn-")]
        if not cands:
            raise FileNotFoundError(f"No mgcnn-* folder under {data_dir} (set --download)")
        cands.sort(key=lambda p: len(p))
        root = cands[0]

    mats = find_files_recursive(root, lambda p: p.lower().endswith(".mat") and dataset in os.path.basename(p).lower())
    if len(mats) == 0:
        mats = find_files_recursive(root, lambda p: p.lower().endswith(".mat") and ("data" in p.lower()))
    if len(mats) == 0:
        raise FileNotFoundError(f"No .mat file found for {dataset} under {root}")

    mat_path = mats[0]
    print(f"[MGCNN] Using MAT: {mat_path}")
    data = load_mat_any(mat_path)

    rating_name, Rmat = _pick_matrix_like(data, ["M", "R", "rating", "ratings", "W", "X"])
    if Rmat is None:
        raise RuntimeError(f"Rating matrix not found in {mat_path}. Keys={list(data.keys())}")

    train_name = None
    test_name = None
    for cand in ["Otraining", "Otrain", "train_mask", "train", "OTrain"]:
        if cand in data:
            train_name = cand
            break
    for cand in ["Otest", "test_mask", "test", "OTest"]:
        if cand in data:
            test_name = cand
            break

    if train_name is None or test_name is None:
        for k, v in data.items():
            if k.startswith("__"):
                continue
            if train_name is None and ("train" in k.lower()):
                train_name = k
            if test_name is None and ("test" in k.lower()):
                test_name = k

    if train_name is None or test_name is None:
        raise RuntimeError(
            f"Train/test mask not found in {mat_path}. "
            f"Found rating='{rating_name}'. Keys={list(data.keys())}"
        )

    Otr = data[train_name]
    Ote = data[test_name]

    import scipy.sparse as sp
    R_csr = _to_csr(Rmat) if _as_scipy_sparse(Rmat) else None
    Otr_csr = _to_csr(Otr) if _as_scipy_sparse(Otr) else None
    Ote_csr = _to_csr(Ote) if _as_scipy_sparse(Ote) else None

    if R_csr is not None:
        num_users, num_items = R_csr.shape
    else:
        Rarr = np.asarray(Rmat)
        if Rarr.ndim != 2:
            raise RuntimeError(f"Rating matrix is not 2D: name={rating_name}, shape={Rarr.shape}")
        num_users, num_items = Rarr.shape

    def _mask_nonzero(mask):
        if sp.issparse(mask):
            coo = mask.tocoo()
            return coo.row.astype(np.int64), coo.col.astype(np.int64)
        m = np.asarray(mask)
        rr, cc = np.nonzero(m)
        return rr.astype(np.int64), cc.astype(np.int64)

    tr_u, tr_i = _mask_nonzero(Otr_csr if Otr_csr is not None else Otr)
    te_u, te_i = _mask_nonzero(Ote_csr if Ote_csr is not None else Ote)

    def _gather_ratings(rr, cc):
        if R_csr is not None:
            vals = R_csr[rr, cc].A1.astype(np.float32)
            return vals
        Rarr2 = np.asarray(Rmat, dtype=np.float32)
        return Rarr2[rr, cc].astype(np.float32)

    tr_r = _gather_ratings(tr_u, tr_i)
    te_r = _gather_ratings(te_u, te_i)

    keep_tr = tr_r != 0
    keep_te = te_r != 0
    tr_u, tr_i, tr_r = tr_u[keep_tr], tr_i[keep_tr], tr_r[keep_tr]
    te_u, te_i, te_r = te_u[keep_te], te_i[keep_te], te_r[keep_te]

    return tr_u, tr_i, tr_r, te_u, te_i, te_r, num_users, num_items


def split_train_into_train_val(u_tr, i_tr, r_tr, val_ratio: float, seed: int):
    n = len(r_tr)
    if val_ratio <= 0 or n == 0:
        return u_tr, i_tr, r_tr, np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(n * val_ratio))
    n_val = max(1, n_val)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    return (
        u_tr[tr_idx], i_tr[tr_idx], r_tr[tr_idx],
        u_tr[val_idx], i_tr[val_idx], r_tr[val_idx],
    )


# -----------------------------
# Models
# -----------------------------
class BasicMF(nn.Module):
    def __init__(self, num_users: int, num_items: int, R: int, mu: float,
                 init_scale: float = 0.02, use_bias: bool = True, device="cpu"):
        super().__init__()
        self.use_bias = use_bias
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float32, device=device), requires_grad=False)
        self.A = nn.Parameter(init_scale * torch.randn(num_users, R, device=device))
        self.B = nn.Parameter(init_scale * torch.randn(num_items, R, device=device))
        if use_bias:
            self.bu = nn.Parameter(torch.zeros(num_users, device=device))
            self.bi = nn.Parameter(torch.zeros(num_items, device=device))

    def predict(self, u: torch.Tensor, it: torch.Tensor) -> torch.Tensor:
        dot = torch.sum(self.A[u] * self.B[it], dim=-1)
        if self.use_bias:
            return self.mu + self.bu[u] + self.bi[it] + dot
        return self.mu + dot


class MMF(nn.Module):
    """
    Shared A/B + K row-wise masks.

    shift_mode:
      - "zero"    : shifts are all zero (fastest). precompute w_k(r) = mA_k(r)*mB_k(r).
      - "random"  : fixed random shifts per (k,row), not learned.
      - "learned" : shifts are trainable nn.Parameters (you can set lr_shifts).
    """

    def __init__(
            self,
            num_users: int,
            num_items: int,
            R: int,
            K: int = 2,
            mu: float = 0.0,
            init_scale: float = 0.02,
            device="cpu",
            mask_mode_A: str = "sin",
            mask_mode_B: str = "sin",
            use_bias: bool = True,
            shift_mode: str = "zero",      # "zero" | "random" | "learned"
            shift_std: float = 0.1,        # used for random init
            shift_init: str = "zero",      # for learned: "zero" | "random"
    ):
        super().__init__()
        assert K >= 2
        assert shift_mode in ["zero", "random", "learned"]
        assert shift_init in ["zero", "random"]

        self.I, self.J, self.R, self.K = num_users, num_items, R, K
        self.half = self.R / 2.0
        self.mask_mode_A = mask_mode_A
        self.mask_mode_B = mask_mode_B
        self.shift_mode = shift_mode
        self.shift_std = float(shift_std)
        self.shift_init = shift_init

        self.use_bias = use_bias
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float32, device=device), requires_grad=False)

        self.A = nn.Parameter(init_scale * torch.randn(num_users, R, device=device))
        self.B = nn.Parameter(init_scale * torch.randn(num_items, R, device=device))

        if use_bias:
            self.bu = nn.Parameter(torch.zeros(num_users, device=device))
            self.bi = nn.Parameter(torch.zeros(num_items, device=device))

        self.register_buffer("pos", torch.arange(R, device=device, dtype=torch.float32))
        self.register_buffer("omega", torch.linspace(1 / K, 1, steps=K, device=device))

        # shifts
        if self.shift_mode == "random":
            self.register_buffer("shiftA_buf", self.shift_std * torch.randn(K, num_users, device=device))
            self.register_buffer("shiftB_buf", self.shift_std * torch.randn(K, num_items, device=device))
        elif self.shift_mode == "learned":
            if shift_init == "random":
                initA = self.shift_std * torch.randn(K, num_users, device=device)
                initB = self.shift_std * torch.randn(K, num_items, device=device)
            else:
                initA = torch.zeros(K, num_users, device=device)
                initB = torch.zeros(K, num_items, device=device)
            self.shiftA = nn.Parameter(initA)  # (K, I)
            self.shiftB = nn.Parameter(initB)  # (K, J)
        else:
            self.register_buffer("w", self._precompute_w(device=device))  # (K, R)

    def _mask_fn(self, delta: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "psin":
            return (torch.sin(math.pi * delta) + 2.5) / 5
        if mode == "pcos":
            return (torch.cos(math.pi * delta) + 2.5) / 5
        if mode == "sin":
            return torch.sin(delta)
        if mode == "cos":
            return torch.cos(delta)
        if mode == "sigmoid":
            return 2.0 * torch.sigmoid(delta)
        if mode == "sinc":
            return (torch.sinc(delta / math.pi) + 1) * 2 / 5
        if mode == "square":
            return torch.tanh(torch.sin(delta) / 0.1) + 1.0
        if mode == "gaussian":
            return 0.5 + torch.exp(-delta ** 2 / 2) / 2.507
        if mode == "cauchy":
            return 2 / (4 + delta ** 2) + 0.5
        if mode == "sin01":
            return (torch.sin(delta) + 1.0) / 2.0
        if mode == "cos01":
            return (torch.cos(delta) + 1.0) / 2.0
        if mode == "triangle":
            return torch.clamp(1.0 - torch.abs(delta), min=0.0)
        raise ValueError(f"Unknown mask mode: {mode}")

    @torch.no_grad()
    def _precompute_w(self, device: str) -> torch.Tensor:
        w_list = []
        pos = self.pos
        for k in range(self.K):
            delta = pos * self.omega[k]
            mA = self._mask_fn(delta, self.mask_mode_A) / self.K
            mB = self._mask_fn(delta, self.mask_mode_B) / self.K
            w_list.append(mA * mB)
        return torch.stack(w_list, dim=0).to(device)

    def _get_shift(self, which: str, k: int, idx: torch.Tensor) -> torch.Tensor:
        """
        Return shift vector (N,) for selected rows.
        """
        if self.shift_mode == "random":
            if which == "A":
                return self.shiftA_buf[k][idx]
            return self.shiftB_buf[k][idx]
        elif self.shift_mode == "learned":
            if which == "A":
                return self.shiftA[k][idx]
            return self.shiftB[k][idx]
        else:
            raise RuntimeError("_get_shift should not be called in shift_mode='zero'")

    def _row_mask_subset(self, idx: torch.Tensor, mode: str, k: int, which: str) -> torch.Tensor:
        """
        idx: (N,) indices (users or items)
        return: (N, R) mask
        """
        u_shift = self._get_shift(which, k, idx)     # (N,)
        s = self.half * u_shift                      # (N,)
        delta = self.pos[None, :] - s[:, None]       # (N, R)
        mask = self._mask_fn(delta * self.omega[k], mode) / self.K
        return mask

    def predict(self, u: torch.Tensor, it: torch.Tensor) -> torch.Tensor:
        A_sel = self.A[u]          # (N, R)
        B_sel = self.B[it]         # (N, R)
        AB = A_sel * B_sel         # (N, R)
        pred = torch.zeros(u.shape[0], device=AB.device, dtype=AB.dtype)

        if self.shift_mode == "zero":
            for k in range(self.K):
                pred = pred + torch.sum(AB * self.w[k], dim=-1)
        else:
            for k in range(self.K):
                mA = self._row_mask_subset(u, self.mask_mode_A, k, which="A")
                mB = self._row_mask_subset(it, self.mask_mode_B, k, which="B")
                pred = pred + torch.sum(AB * (mA * mB), dim=-1)

        if self.use_bias:
            pred = pred + self.mu + self.bu[u] + self.bi[it]
        else:
            pred = pred + self.mu
        return pred


# -----------------------------
# Training / Eval
# -----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str,
             rating_min: float, rating_max: float) -> Tuple[float, float]:
    model.eval()
    preds = []
    ys = []
    for u, it, y in loader:
        u = u.to(device)
        it = it.to(device)
        y = y.to(device)
        p = model.predict(u, it)
        p = torch.clamp(p, rating_min, rating_max)
        preds.append(p)
        ys.append(y)
    pred = torch.cat(preds, dim=0) if len(preds) else torch.empty(0, device=device)
    y = torch.cat(ys, dim=0) if len(ys) else torch.empty(0, device=device)
    if pred.numel() == 0:
        return float("nan"), float("nan")
    return rmse_mae(pred, y)


@dataclass
class TrainResult:
    best_epoch: int
    best_val_rmse: float
    test_rmse: float
    test_mae: float
    train_time_sec: float
    history: Dict[str, list]


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: str,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        patience: int,
        print_every: int,
        rating_min: float,
        rating_max: float,
) -> TrainResult:
    best_state = None
    best_epoch = -1
    best_val_rmse = float("inf")
    bad = 0
    history = {"val_rmse": [], "val_mae": []}

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for u, it, y in train_loader:
            u = u.to(device)
            it = it.to(device)
            y = y.to(device)

            pred = model.predict(u, it)
            loss = torch.mean((pred - y) ** 2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if len(val_loader.dataset) == 0:
            val_rmse, val_mae = float("inf"), float("inf")
            improved = True
        else:
            val_rmse, val_mae = evaluate(model, val_loader, device, rating_min, rating_max)
            improved = val_rmse < best_val_rmse - 1e-6

        history["val_rmse"].append(val_rmse)
        history["val_mae"].append(val_mae)

        if improved:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = cpu_state_dict_clone(model)
            bad = 0
        else:
            bad += 1

        if epoch % print_every == 0:
            print(f"[Epoch {epoch:3d}/{epochs}] val_RMSE={val_rmse:.4f} val_MAE={val_mae:.4f} "
                  f"(best_RMSE={best_val_rmse:.4f}@{best_epoch}, bad={bad}/{patience})")

        if len(val_loader.dataset) > 0 and bad >= patience:
            print(f"[EarlyStop] no improvement for {patience} epochs.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_rmse, test_mae = evaluate(model, test_loader, device, rating_min, rating_max)
    t1 = time.time()

    return TrainResult(
        best_epoch=best_epoch,
        best_val_rmse=best_val_rmse,
        test_rmse=test_rmse,
        test_mae=test_mae,
        train_time_sec=(t1 - t0),
        history=history,
    )


def make_loaders(u_tr, i_tr, r_tr, u_va, i_va, r_va, u_te, i_te, r_te,
                 batch_size: int, num_workers: int = 0, pin_memory: bool = False):
    def to_tensor(x, dtype): return torch.tensor(x, dtype=dtype)

    train_ds = TensorDataset(to_tensor(u_tr, torch.long), to_tensor(i_tr, torch.long), to_tensor(r_tr, torch.float32))
    val_ds = TensorDataset(to_tensor(u_va, torch.long), to_tensor(i_va, torch.long), to_tensor(r_va, torch.float32))
    test_ds = TensorDataset(to_tensor(u_te, torch.long), to_tensor(i_te, torch.long), to_tensor(r_te, torch.float32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


def plot_curves(save_path: str, base_hist: Dict[str, list], mmf_hist: Dict[str, list], title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(base_hist["val_rmse"], label="Baseline MF (val RMSE)")
    plt.plot(mmf_hist["val_rmse"], label="MMF (val RMSE)")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# -----------------------------
# Dataset abstraction
# -----------------------------
@dataclass
class SplitPack:
    u_tr: np.ndarray
    i_tr: np.ndarray
    r_tr: np.ndarray
    u_va: np.ndarray
    i_va: np.ndarray
    r_va: np.ndarray
    u_te: np.ndarray
    i_te: np.ndarray
    r_te: np.ndarray
    num_users: int
    num_items: int
    rating_min: float
    rating_max: float


def get_splits(args, seed: int) -> SplitPack:
    ds = args.dataset

    if ds in ["ml-100k", "ml-1m", "ml-10m"]:
        u_all, i_all, r_all, num_users, num_items = load_movielens_ratings(ds, args.data_dir, download=args.download)
        (u_tr, i_tr, r_tr,
         u_va, i_va, r_va,
         u_te, i_te, r_te) = per_user_split(
            u_all, i_all, r_all, num_users=num_users,
            val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=seed
        )
        rating_min = float(np.min(r_all)) if args.rating_min is None else float(args.rating_min)
        rating_max = float(np.max(r_all)) if args.rating_max is None else float(args.rating_max)
        return SplitPack(u_tr, i_tr, r_tr, u_va, i_va, r_va, u_te, i_te, r_te,
                         num_users, num_items, rating_min, rating_max)

    if ds in ["flixster", "douban"]:
        tr_u, tr_i, tr_r, te_u, te_i, te_r, num_users, num_items = load_flixster_douban_from_mgcnn(
            ds, args.data_dir, download=args.download
        )
        u_tr, i_tr, r_tr, u_va, i_va, r_va = split_train_into_train_val(
            tr_u, tr_i, tr_r, val_ratio=args.val_ratio, seed=seed
        )
        u_te, i_te, r_te = te_u, te_i, te_r

        all_r = np.concatenate([tr_r, te_r], axis=0) if len(te_r) else tr_r
        rating_min = float(np.min(all_r)) if args.rating_min is None else float(args.rating_min)
        rating_max = float(np.max(all_r)) if args.rating_max is None else float(args.rating_max)
        return SplitPack(u_tr, i_tr, r_tr, u_va, i_va, r_va, u_te, i_te, r_te,
                         num_users, num_items, rating_min, rating_max)

    raise ValueError(f"Unknown dataset: {ds}")


# -----------------------------
# Main
# -----------------------------
def main():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, default="ml-1m",
                   choices=["ml-100k", "ml-1m", "ml-10m", "flixster", "douban"])
    p.add_argument("--data_dir", type=str, default="./data_rec")
    p.add_argument("--download", action="store_true", help="download dataset if not present")

    # multi-seed options
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--seed_start", type=int, default=123)
    p.add_argument("--num_seeds", type=int, default=1)

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--plot_curves", action="store_true",
                   help="Save validation RMSE curves as a PNG (disabled by default).")

    # split
    p.add_argument("--val_ratio", type=float, default=0.001)
    p.add_argument("--test_ratio", type=float, default=0.1)

    # rating clamp (optional override)
    p.add_argument("--rating_min", type=float, default=None)
    p.add_argument("--rating_max", type=float, default=None)

    # model capacity
    p.add_argument("--R", type=int, default=1200)
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--use_bias", dest="use_bias", action="store_true",
                   help="Enable user/item biases (default: enabled).")
    p.add_argument("--no_bias", dest="use_bias", action="store_false",
                   help="Disable user/item biases.")
    p.set_defaults(use_bias=True)

    # mmf masks
    p.add_argument("--mask_mode_A", type=str, default="gaussian")
    p.add_argument("--mask_mode_B", type=str, default="gaussian")

    # optimization
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers (0 is safest for reproducibility).")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--print_every", type=int, default=1)

    p.add_argument("--lr_baseline", type=float, default=2e-3)
    p.add_argument("--lr_factors", type=float, default=5e-3)
    p.add_argument("--lr_shifts", type=float, default=5e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)

    # shift settings
    p.add_argument("--shift_mode", type=str, default="zero",
                   choices=["zero", "random", "learned"],
                   help="How to handle shift parameters in MMF.")
    p.add_argument("--shift_init", type=str, default="zero", choices=["zero", "random"])  # used when learned
    p.add_argument("--shift_std", type=float, default=0.1)  # used for random init

    # output
    p.add_argument("--save_dir", type=str, default="./results_mmf")
    args = p.parse_args()

    ensure_dir(args.save_dir)

    if args.seeds is not None and len(args.seeds) > 0:
        seed_list = list(args.seeds)
    else:
        seed_list = list(range(args.seed_start, args.seed_start + args.num_seeds))

    print(f"[Run] dataset={args.dataset} seeds={seed_list} device={args.device} "
          f"R={args.R} K={args.K} val_ratio={args.val_ratio} use_bias={args.use_bias} "
          f"shift_mode={args.shift_mode}")

    all_summaries = []

    for seed in seed_list:
        print("\n" + "=" * 70)
        print(f"[Seed {seed}] start")
        print("=" * 70)
        set_seed(seed)

        pack = get_splits(args, seed)
        mu = float(np.mean(pack.r_tr)) if len(pack.r_tr) else 0.0

        print(f"[Data] users={pack.num_users:,} items={pack.num_items:,} "
              f"train={len(pack.r_tr):,} val={len(pack.r_va):,} test={len(pack.r_te):,} "
              f"mu(train)={mu:.4f} clamp=[{pack.rating_min:.2f},{pack.rating_max:.2f}]")

        train_loader, val_loader, test_loader = make_loaders(
            pack.u_tr, pack.i_tr, pack.r_tr,
            pack.u_va, pack.i_va, pack.r_va,
            pack.u_te, pack.i_te, pack.r_te,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=str(args.device).startswith("cuda")
        )

        device = args.device

        # ---- Baseline MF ----
        base = BasicMF(pack.num_users, pack.num_items, R=args.R, mu=3.0,
                       use_bias=args.use_bias, device=device).to(device)
        opt_base = torch.optim.Adam(base.parameters(), lr=args.lr_baseline, weight_decay=args.weight_decay)
        print(f"\n[Baseline MF] params={count_params(base):,}")
        res_base = train_model(
            base, train_loader, val_loader, test_loader,
            device=device, epochs=args.epochs, optimizer=opt_base,
            patience=args.patience, print_every=args.print_every,
            rating_min=pack.rating_min, rating_max=pack.rating_max
        )

        # ---- MMF ----
        mmf = MMF(
            pack.num_users, pack.num_items, R=args.R, K=args.K, mu=3.0,
            mask_mode_A=args.mask_mode_A, mask_mode_B=args.mask_mode_B,
            use_bias=args.use_bias, device=device,
            shift_mode=args.shift_mode, shift_std=args.shift_std, shift_init=args.shift_init,
        ).to(device)

        param_groups = [{
            "name": "factors",
            "params": [mmf.A, mmf.B] + ([mmf.bu, mmf.bi] if args.use_bias else []),
            "lr": args.lr_factors,
            "weight_decay": args.weight_decay
        }]

        if args.shift_mode == "learned":
            param_groups.append({
                "name": "shifts",
                "params": [mmf.shiftA, mmf.shiftB],
                "lr": args.lr_shifts,
                "weight_decay": 0.0
            })

        opt_mmf = torch.optim.Adam(param_groups)

        extra = ""
        if args.shift_mode == "learned":
            extra = f" shift_init={args.shift_init} lr_shifts={args.lr_shifts:g}"
        elif args.shift_mode == "random":
            extra = f" shift_std={args.shift_std:g} (fixed)"

        print(f"\n[MMF] params={count_params(mmf):,} maskA={args.mask_mode_A} maskB={args.mask_mode_B} "
              f"shift={args.shift_mode}{extra}")

        res_mmf = train_model(
            mmf, train_loader, val_loader, test_loader,
            device=device, epochs=args.epochs, optimizer=opt_mmf,
            patience=args.patience, print_every=args.print_every,
            rating_min=pack.rating_min, rating_max=pack.rating_max
        )

        summary = {
            "dataset": args.dataset,
            "seed": seed,
            "num_users": int(pack.num_users),
            "num_items": int(pack.num_items),
            "split": {"train": int(len(pack.r_tr)), "val": int(len(pack.r_va)), "test": int(len(pack.r_te))},
            "rating_clamp": {"min": pack.rating_min, "max": pack.rating_max},
            "config": {
                "R": args.R, "K": args.K, "use_bias": args.use_bias,
                "mask_mode_A": args.mask_mode_A, "mask_mode_B": args.mask_mode_B,
                "shift_mode": args.shift_mode, "shift_init": args.shift_init, "shift_std": args.shift_std,
                "lr_shifts": args.lr_shifts,
                "epochs": args.epochs, "batch_size": args.batch_size,
                "lr_baseline": args.lr_baseline, "lr_factors": args.lr_factors,
                "patience": args.patience,
                "device": args.device,
            },
            "baseline": {
                "params": count_params(base),
                "best_epoch": res_base.best_epoch,
                "best_val_rmse": res_base.best_val_rmse,
                "test_rmse": res_base.test_rmse,
                "test_mae": res_base.test_mae,
                "train_time_sec": res_base.train_time_sec,
            },
            "mmf": {
                "params": count_params(mmf),
                "best_epoch": res_mmf.best_epoch,
                "best_val_rmse": res_mmf.best_val_rmse,
                "test_rmse": res_mmf.test_rmse,
                "test_mae": res_mmf.test_mae,
                "train_time_sec": res_mmf.train_time_sec,
            },
        }

        print("\n==================== FINAL COMPARISON (seed) ====================")
        print(f"Dataset: {args.dataset} | seed={seed} | bias={args.use_bias}")
        print(f"Baseline MF: params={summary['baseline']['params']:,}  "
              f"best@{summary['baseline']['best_epoch']}  "
              f"test_RMSE={summary['baseline']['test_rmse']:.4f}  test_MAE={summary['baseline']['test_mae']:.4f}  "
              f"time={summary['baseline']['train_time_sec']:.1f}s")
        print(f"MMF:        params={summary['mmf']['params']:,}  "
              f"best@{summary['mmf']['best_epoch']}  "
              f"test_RMSE={summary['mmf']['test_rmse']:.4f}  test_MAE={summary['mmf']['test_mae']:.4f}  "
              f"time={summary['mmf']['train_time_sec']:.1f}s")
        print("===============================================================\n")

        out_json = os.path.join(args.save_dir, f"{args.dataset}_R{args.R}_K{args.K}_seed{seed}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[Saved] {out_json}")

        if args.plot_curves:
            out_png = os.path.join(args.save_dir, f"{args.dataset}_curves_R{args.R}_K{args.K}_seed{seed}.png")
            plot_curves(out_png, res_base.history, res_mmf.history,
                        title=f"{args.dataset} | R={args.R} | K={args.K} | bias={args.use_bias} | seed={seed}")
            print(f"[Saved] {out_png}")

        all_summaries.append(summary)

        del base, mmf, opt_base, opt_mmf
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # aggregate
    def _mean_std(xs):
        xs = np.asarray(xs, dtype=float)
        return float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0

    base_rmse = [s["baseline"]["test_rmse"] for s in all_summaries]
    mmf_rmse = [s["mmf"]["test_rmse"] for s in all_summaries]
    base_mae = [s["baseline"]["test_mae"] for s in all_summaries]
    mmf_mae = [s["mmf"]["test_mae"] for s in all_summaries]
    base_time = [s["baseline"]["train_time_sec"] for s in all_summaries]
    mmf_time = [s["mmf"]["train_time_sec"] for s in all_summaries]

    b_rmse_m, b_rmse_s = _mean_std(base_rmse)
    m_rmse_m, m_rmse_s = _mean_std(mmf_rmse)
    b_mae_m, b_mae_s = _mean_std(base_mae)
    m_mae_m, m_mae_s = _mean_std(mmf_mae)
    tr_m, tr_s = _mean_std([m / b for m, b in zip(mmf_time, base_time)])

    agg = {
        "dataset": args.dataset,
        "seeds": seed_list,
        "baseline": {"test_rmse_mean": b_rmse_m, "test_rmse_std": b_rmse_s,
                     "test_mae_mean": b_mae_m, "test_mae_std": b_mae_s,
                     "time_sec_mean": float(np.mean(base_time)),
                     "time_sec_std": float(np.std(base_time, ddof=1)) if len(base_time) > 1 else 0.0},
        "mmf": {"test_rmse_mean": m_rmse_m, "test_rmse_std": m_rmse_s,
                "test_mae_mean": m_mae_m, "test_mae_std": m_mae_s,
                "time_sec_mean": float(np.mean(mmf_time)),
                "time_sec_std": float(np.std(mmf_time, ddof=1)) if len(mmf_time) > 1 else 0.0},
        "time_ratio(mmf/baseline)": {"mean": tr_m, "std": tr_s},
    }

    print("\n" + "=" * 70)
    print("[Aggregate over seeds]")
    print(f"Seeds: {seed_list}")
    print(f"Baseline test RMSE: {b_rmse_m:.4f} ± {b_rmse_s:.4f}")
    print(f"MMF      test RMSE: {m_rmse_m:.4f} ± {m_rmse_s:.4f}")
    print(f"Baseline test MAE : {b_mae_m:.4f} ± {b_mae_s:.4f}")
    print(f"MMF      test MAE : {m_mae_m:.4f} ± {m_mae_s:.4f}")
    print(f"Time ratio (mmf/base): {tr_m:.2f} ± {tr_s:.2f}")
    print("=" * 70)

    out_agg = os.path.join(args.save_dir, f"{args.dataset}_R{args.R}_K{args.K}_agg.json")
    with open(out_agg, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    print(f"[Saved] {out_agg}")


if __name__ == "__main__":
    main()