#!/usr/bin/env python3
"""
A-sweep ablation backing `tab:A-sweep` and `tab:mutual-union` (app:A-sweep
in PAPER/main.tex).

Sweeps the finite-sample logarithmic coefficient A in
    k_star = ceil(A * log(4n/delta))
across 7 values on representative synthetic datasets, plus an extra
mutual-vs-union ablation at A=1.

Output: results/ablations/A_sweep.csv with columns
    dataset, K_true, A_coef, candidate_mode, seed,
    k_star, K_low, K_high, K_hat, K_prac, ARI, regime,
    rho_hat, n_active, n_edges_final, runtime
"""
from __future__ import annotations
import csv
import os
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from MBC import MBCParams, mbc_cluster, get_labels_for_K  # noqa: E402


# -----------------------------------------------------------------------------
# Generators (lifted minimally from run_synth.py)
# -----------------------------------------------------------------------------
def gen_blobs_clean(n, seed):
    X, y = make_blobs(n_samples=n, centers=4, n_features=2,
                      cluster_std=0.9, random_state=seed)
    return X, y


def gen_moons_noisy(n, seed, sigma=0.15):
    X, y = make_moons(n_samples=n, noise=sigma, random_state=seed)
    return X, y


def gen_blobs_bg(n, seed, frac=0.10):
    X, y = make_blobs(n_samples=int(n * (1 - frac)), centers=4,
                      n_features=2, cluster_std=0.9, random_state=seed)
    rng = np.random.default_rng(seed)
    lo, hi = X.min(axis=0), X.max(axis=0)
    pad = 0.1 * (hi - lo + 1e-9)
    n_noise = max(1, int(frac * len(X) / max(1e-9, 1.0 - frac)))
    noise = rng.uniform(lo - pad, hi + pad, size=(n_noise, X.shape[1]))
    Xa = np.vstack([X, noise])
    ya = np.concatenate([y, np.full(n_noise, -1, dtype=int)])
    perm = rng.permutation(len(Xa))
    return Xa[perm], ya[perm]


def gen_blobs_50d(n, seed):
    X, y = make_blobs(n_samples=n, centers=6, n_features=50,
                      cluster_std=2.0, random_state=seed)
    return X, y


def gen_hier_2d(n, seed):
    """3x3 hierarchical blob grid in 2D."""
    rng = np.random.default_rng(seed)
    per = max(1, n // 9)
    Xs, ys = [], []
    macro_sep = 12.0
    micro_sep = 2.0
    label = 0
    for i in range(3):
        for j in range(3):
            cx = i * macro_sep + (i % 3) * micro_sep
            cy = j * macro_sep + (j % 3) * micro_sep
            Xs.append(rng.normal(loc=(cx, cy), scale=0.4, size=(per, 2)))
            ys.append(np.full(per, label, dtype=int))
            label += 1
    return np.vstack(Xs), np.concatenate(ys)


DATASETS = [
    # (name, K_true, generator, regime_class)
    ("blobs_2D_clean",  4, gen_blobs_clean,  "separable"),
    ("blobs_2D_bg10",   4, gen_blobs_bg,     "transitional (contam.)"),
    ("moons_n0.15",     2, gen_moons_noisy,  "non-separable (noise)"),
    ("blobs_50D_easy",  6, gen_blobs_50d,    "high-D separable"),
    ("hier_3x3_2D",     9, gen_hier_2d,      "multi-scale"),
]

A_VALUES = [0.5, 1.0, 1.5, 2.0, 4.0, 8.0, 12.0]
SEEDS = [7, 11, 23]
N_SAMPLES = 2000


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
def run_one(name, K_true, X, y, A_coef, candidate_mode, seed):
    """Run MBC with given A_coef + candidate_mode and return a row dict."""
    Xs = StandardScaler().fit_transform(X)
    p = MBCParams(
        delta=0.05,
        A_coef=A_coef,
        candidate_mode=candidate_mode,
    )
    t0 = time.time()
    res = mbc_cluster(Xs, p, seed=seed)
    runtime = time.time() - t0

    # Score against ground truth (drop background label -1 if present)
    mask = y >= 0
    ari = float("nan")
    if mask.sum() > 1:
        try:
            ari = adjusted_rand_score(y[mask], res.labels[mask])
        except Exception:
            ari = float("nan")

    info = res.info or {}
    K_low_b, K_high_b = res.bracket
    mass_b = info.get("bracket_mass", (-1, -1))
    return {
        "dataset": name,
        "K_true": K_true,
        "A_coef": A_coef,
        "candidate_mode": candidate_mode,
        "seed": seed,
        "k_star": int(res.k_star),
        "k_low": int(res.k_low),
        "k_high": int(res.k_high),
        "K_low": int(K_low_b),
        "K_high": int(K_high_b),
        "K_low_mass": int(mass_b[0]),
        "K_high_mass": int(mass_b[1]),
        "K_hat_strict": int(info.get("K_persistent_strict", -1)),
        "K_prac": int(res.n_clusters),
        "ARI": ari,
        "regime": info.get("regime", ""),
        "rho_hat": float(res.rho_hat),
        "n_active": int(info.get("n_active", -1)),
        "n_edges_final": int(res.n_edges_final),
        "runtime": runtime,
    }


def main():
    out_dir = ROOT / "results" / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "A_sweep.csv"

    rows = []
    total = len(DATASETS) * len(A_VALUES) * len(SEEDS) + len(DATASETS) * len(SEEDS)
    done = 0
    for name, K_true, gen, regime_class in DATASETS:
        for seed in SEEDS:
            X, y = gen(N_SAMPLES, seed)
            # A-sweep with mutual graph
            for A in A_VALUES:
                done += 1
                row = run_one(name, K_true, X, y, A, "mutual", seed)
                row["regime_class"] = regime_class
                rows.append(row)
                print(f"[{done}/{total}] {name:20s} A={A:5.1f} mutual seed={seed} "
                      f"k*={row['k_star']:3d} k=[{row['k_low']},{row['k_high']}] "
                      f"K=[{row['K_low']},{row['K_high']}] "
                      f"K^={row['K_prac']} rho={row['rho_hat']:.2f} "
                      f"reg={row['regime']:12s} ARI={row['ARI']:.3f}")
            # union graph at A=1 only (mutual-vs-union ablation)
            done += 1
            row = run_one(name, K_true, X, y, 1.0, "union", seed)
            row["regime_class"] = regime_class
            rows.append(row)
            print(f"[{done}/{total}] {name:20s} A=  1.0 union  seed={seed} "
                  f"k*={row['k_star']:3d} k=[{row['k_low']},{row['k_high']}] "
                  f"K=[{row['K_low']},{row['K_high']}] "
                  f"K^={row['K_prac']} rho={row['rho_hat']:.2f} "
                  f"reg={row['regime']:12s} ARI={row['ARI']:.3f}")

    fieldnames = [
        "dataset", "regime_class", "K_true", "A_coef", "candidate_mode", "seed",
        "k_star", "k_low", "k_high",
        "K_low", "K_high", "K_low_mass", "K_high_mass",
        "K_hat_strict", "K_prac", "ARI",
        "regime", "rho_hat", "n_active", "n_edges_final", "runtime",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
