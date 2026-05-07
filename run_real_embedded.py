#!/usr/bin/env python3
"""
run_real_embedded.py — test MBC brackets vs DBSCAN/HDBSCAN grids on the
real-world suite *after* embedding into the data's intrinsic dimension.

For each of the 11 datasets in run_real.py we:
  1. Estimate intrinsic dimension d_int from per-point Levina-Bickel MLE
     (k=10), reporting both max and 95th-percentile and capping the
     embedding target at min(d_max, ambient D, 32).
  2. Embed in R^{d_target} via four methods: PCA, UMAP, Isomap,
     diffusion maps (sparse-kNN Gaussian kernel).
  3. Run MBC, DBSCAN-grid (6 configs), HDBSCAN-grid (8 configs) on each
     embedding and record bracket / K_hat / ARI per (dataset, embed,
     algo, seed).

Output:
  results/real_embedded/real_embedded_raw.csv   — all rows
  results/real_embedded/real_embedded_summary.csv  — per (dataset, embed,
                                                       algo) 3-seed mean
"""
from __future__ import annotations

import argparse
import math
import time
import warnings
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

from MBC import MBCParams, mbc_cluster
from run_real import real_specs

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


# =============================================================================
# Intrinsic-dimension estimator (Levina-Bickel MLE, per point)
# =============================================================================

def local_intrinsic_dim(X: np.ndarray, k: int = 10) -> np.ndarray:
    """Per-point Levina-Bickel MLE local intrinsic dimension.

    For each point, ID = (1/(k-1)) sum_j log(r_k / r_j) for j=1..k-1; we
    return 1 / that, which is the per-point d-hat.
    """
    n = X.shape[0]
    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dists, _ = nn.kneighbors(X)
    R = np.maximum(dists[:, 1:], 1e-12)  # (n, k)
    log_ratios = np.log(R[:, -1:] / R[:, :-1])  # (n, k-1)
    inv_d = log_ratios.mean(axis=1)
    inv_d = np.maximum(inv_d, 1e-12)
    return 1.0 / inv_d


# =============================================================================
# Embeddings
# =============================================================================

def pca_embed(X: np.ndarray, d: int, *, seed: int = 0) -> np.ndarray:
    d = min(d, X.shape[1], X.shape[0] - 1)
    return PCA(n_components=d, random_state=seed).fit_transform(X)


def umap_embed(X: np.ndarray, d: int, *, seed: int = 0) -> Optional[np.ndarray]:
    if not UMAP_AVAILABLE:
        return None
    d = min(d, X.shape[1])
    n = X.shape[0]
    n_nb = min(15, n - 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer = umap.UMAP(n_components=d, random_state=seed,
                            n_neighbors=n_nb, min_dist=0.1)
        return reducer.fit_transform(X)


def isomap_embed(X: np.ndarray, d: int, *, seed: int = 0) -> Optional[np.ndarray]:
    n = X.shape[0]
    d = min(d, X.shape[1])
    n_nb = min(10, n - 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return Isomap(n_neighbors=n_nb, n_components=d).fit_transform(X)
        except Exception as e:
            print(f"    [isomap] failed: {e}")
            return None


def diffusion_map_embed(X: np.ndarray, d: int, *, seed: int = 0,
                        k_nn: int = 30,
                        sigma_scale: float = 1.0) -> Optional[np.ndarray]:
    """Sparse-kNN diffusion map embedding."""
    n = X.shape[0]
    d = min(d, n - 2)
    k_nn = min(k_nn, n - 1)
    nn = NearestNeighbors(n_neighbors=k_nn + 1).fit(X)
    dists, idx = nn.kneighbors(X)

    sigma = sigma_scale * float(np.median(dists[:, 1:]))
    if sigma <= 0:
        return None
    rows = np.repeat(np.arange(n), k_nn)
    cols = idx[:, 1:].ravel()
    vals = np.exp(-dists[:, 1:].ravel() ** 2 / (2 * sigma ** 2))
    K = csr_matrix((vals, (rows, cols)), shape=(n, n))
    K = (K + K.T) / 2  # symmetrize

    deg = np.array(K.sum(axis=1)).flatten()
    deg = np.maximum(deg, 1e-12)
    D_inv_sqrt = diags(1.0 / np.sqrt(deg))
    Ps = D_inv_sqrt @ K @ D_inv_sqrt

    try:
        eigvals, eigvecs = eigsh(Ps, k=d + 1, which="LM")
    except Exception as e:
        print(f"    [diffusion] eigsh failed: {e}")
        return None
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    eigvals = eigvals[order]
    # Drop the trivial first eigenvector (constant), scale by eigenvalues
    psi = D_inv_sqrt @ eigvecs[:, 1:d + 1]
    psi = np.asarray(psi) * eigvals[1:d + 1]
    return psi


# =============================================================================
# DBSCAN / HDBSCAN grids
# =============================================================================

def kdist_eps(X: np.ndarray, k: int) -> float:
    n = X.shape[0]
    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    d, _ = nn.kneighbors(X)
    return float(np.median(d[:, k]))


def run_dbscan_grid(X: np.ndarray) -> List[Tuple[str, int, np.ndarray]]:
    n = X.shape[0]
    out: List[Tuple[str, int, np.ndarray]] = []
    try:
        k_eps = max(5, int(5 * math.log(max(3, n))))
        eps0 = kdist_eps(X, k_eps)
    except Exception as e:
        print(f"    [DBSCAN grid] kdist failed: {e}")
        return out
    for f in (0.7, 1.0, 1.5):
        for ms in (5, k_eps):
            try:
                labels = DBSCAN(eps=max(1e-9, f * eps0),
                                min_samples=ms).fit_predict(X)
                K = len(set(labels)) - (1 if -1 in labels else 0)
                out.append((f"DBSCAN(eps={f:.1f}*kd,ms={ms})", K, labels))
            except Exception as e:
                print(f"    [DBSCAN(f={f},ms={ms})] {e}")
    return out


def run_hdbscan_grid(X: np.ndarray) -> List[Tuple[str, int, np.ndarray]]:
    if not HDBSCAN_AVAILABLE:
        return []
    n = X.shape[0]
    out: List[Tuple[str, int, np.ndarray]] = []
    for method in ("eom", "leaf"):
        for frac in (0.005, 0.01, 0.02, 0.05):
            mcs = max(5, int(frac * n))
            try:
                labels = hdbscan.HDBSCAN(min_cluster_size=mcs,
                                         cluster_selection_method=method
                                         ).fit_predict(X)
                K = len(set(labels)) - (1 if -1 in labels else 0)
                out.append((f"HDBSCAN({method},mcs={frac:.1%})", K, labels))
            except Exception as e:
                print(f"    [HDBSCAN({method},mcs={frac})] {e}")
    return out


# =============================================================================
# Driver
# =============================================================================

def evaluate_dataset(X: np.ndarray, y: np.ndarray, *,
                     spec_name: str, K_true: int,
                     embedding: str, target_dim: int,
                     d_max: float, d_p95: float, d_med: float,
                     seed: int) -> List[dict]:
    rows = []
    base = dict(
        dataset=spec_name, seed=seed, K_true=K_true,
        n=int(X.shape[0]), D=int(X.shape[1]),
        embedding=embedding, target_dim=target_dim,
        d_max=float(d_max), d_p95=float(d_p95), d_med=float(d_med),
    )

    # MBC
    t0 = time.time()
    res = mbc_cluster(X, MBCParams(), seed=seed)
    runtime = time.time() - t0
    ari = adjusted_rand_score(y, res.labels) if y is not None else float("nan")
    nmi = normalized_mutual_info_score(y, res.labels) if y is not None else float("nan")
    rows.append({**base,
                 "algo": "MBC",
                 "K": int(res.n_clusters),
                 "bracket_low": int(res.bracket[0]),
                 "bracket_high": int(res.bracket[1]),
                 "rho_hat": float(res.rho_hat),
                 "ARI": float(ari), "NMI": float(nmi),
                 "runtime": float(runtime)})
    print(f"    MBC          K={res.n_clusters}  bracket=[{res.bracket[0]},{res.bracket[1]}]  "
          f"rho={res.rho_hat:.2f}  ARI={ari:.3f}")

    # DBSCAN grid
    t0 = time.time()
    dbs = run_dbscan_grid(X)
    runtime = time.time() - t0
    if dbs:
        Ks = [K for _, K, _ in dbs]
        aris = [adjusted_rand_score(y, lab) if y is not None else float("nan")
                for _, _, lab in dbs]
        Kmin, Kmax = int(min(Ks)), int(max(Ks))
        best_ari = float(np.nanmax(aris))
        rows.append({**base,
                     "algo": "DBSCAN-grid",
                     "K": int(np.median(Ks)),
                     "bracket_low": Kmin, "bracket_high": Kmax,
                     "rho_hat": float("nan"),
                     "ARI": best_ari, "NMI": float("nan"),
                     "runtime": float(runtime)})
        print(f"    DBSCAN-grid  K_med={int(np.median(Ks))}  bracket=[{Kmin},{Kmax}]  "
              f"best ARI={best_ari:.3f}")

    # HDBSCAN grid
    t0 = time.time()
    hdb = run_hdbscan_grid(X)
    runtime = time.time() - t0
    if hdb:
        Ks = [K for _, K, _ in hdb]
        aris = [adjusted_rand_score(y, lab) if y is not None else float("nan")
                for _, _, lab in hdb]
        Kmin, Kmax = int(min(Ks)), int(max(Ks))
        best_ari = float(np.nanmax(aris))
        rows.append({**base,
                     "algo": "HDBSCAN-grid",
                     "K": int(np.median(Ks)),
                     "bracket_low": Kmin, "bracket_high": Kmax,
                     "rho_hat": float("nan"),
                     "ARI": best_ari, "NMI": float("nan"),
                     "runtime": float(runtime)})
        print(f"    HDBSCAN-grid K_med={int(np.median(Ks))}  bracket=[{Kmin},{Kmax}]  "
              f"best ARI={best_ari:.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default="results/real_embedded")
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 23])
    ap.add_argument("--max_n", type=int, default=8000)
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Filter to these dataset names (default: all 11).")
    ap.add_argument("--embeddings", nargs="*",
                    default=["pca", "umap", "isomap", "diffusion"])
    ap.add_argument("--id_k", type=int, default=10,
                    help="k for Levina-Bickel ID estimator")
    ap.add_argument("--target_dim_cap", type=int, default=32,
                    help="Cap embedding target dim at min(d_max, D, this)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = real_specs(args.max_n)
    if args.datasets:
        specs = [s for s in specs if s.name in args.datasets]
    print(f"Real-embedded suite: {len(specs)} datasets x "
          f"{len(args.seeds)} seeds x {len(args.embeddings)} embeddings\n")

    embedders = {
        "pca": pca_embed, "umap": umap_embed,
        "isomap": isomap_embed, "diffusion": diffusion_map_embed,
    }

    rows: List[dict] = []
    for spec in specs:
        for seed in args.seeds:
            data = spec.loader(seed)
            if data is None:
                print(f"=== {spec.name} seed={seed}: not available, skipping ===")
                continue
            X, y = data
            n, D = X.shape
            print(f"\n=== {spec.name} seed={seed} (n={n}, D={D}, K_true={spec.K_true}) ===")

            d_per = local_intrinsic_dim(X, k=args.id_k)
            d_max = float(d_per.max())
            d_p95 = float(np.percentile(d_per, 95))
            d_med = float(np.median(d_per))
            d_target = int(min(max(2, math.ceil(d_max)), D, args.target_dim_cap))
            print(f"  intrinsic dim: max={d_max:.1f}, p95={d_p95:.1f}, "
                  f"median={d_med:.1f}, D={D}  ->  target_dim={d_target}")

            for emb_name in args.embeddings:
                t0 = time.time()
                emb_fn = embedders[emb_name]
                X_emb = emb_fn(X, d_target, seed=seed)
                emb_runtime = time.time() - t0
                if X_emb is None:
                    print(f"  embedding={emb_name}: unavailable, skipping")
                    continue
                print(f"  embedding={emb_name}  shape={X_emb.shape}  "
                      f"({emb_runtime:.1f}s)")
                rows.extend(evaluate_dataset(
                    X_emb, y,
                    spec_name=spec.name, K_true=spec.K_true,
                    embedding=emb_name, target_dim=d_target,
                    d_max=d_max, d_p95=d_p95, d_med=d_med,
                    seed=seed,
                ))

    df = pd.DataFrame(rows)
    raw = out_dir / "real_embedded_raw.csv"
    df.to_csv(raw, index=False)
    print(f"\nWrote {raw} ({len(df)} rows)")

    if df.empty:
        return
    summary = df.groupby(["dataset", "embedding", "algo"]).agg(
        n=("n", "first"), D=("D", "first"), K_true=("K_true", "first"),
        target_dim=("target_dim", "first"),
        d_max=("d_max", "mean"), d_p95=("d_p95", "mean"),
        K=("K", "median"),
        bracket_low=("bracket_low", "median"),
        bracket_high=("bracket_high", "median"),
        ARI_mean=("ARI", "mean"), ARI_std=("ARI", "std"),
        runtime=("runtime", "mean"),
    ).reset_index()
    sum_path = out_dir / "real_embedded_summary.csv"
    summary.to_csv(sum_path, index=False)
    print(f"Wrote {sum_path}")


if __name__ == "__main__":
    main()
