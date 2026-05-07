#!/usr/bin/env python3
"""
run_sampling_sweep.py — backs `tab:sampling-sweep` (app:sampling-sweep in PAPER/main.tex).

Sweep n in {200, 500, 1000, 2000, 5000} on three datasets:
  - 2D blobs (sigma=0.9, K=4)
  - two moons (noise=0.10, K=2)
  - 50D blobs easy (K=6, std=1.0)

Three seeds per cell (7, 11, 23). Output:
  results/sampling/sampling_sweep.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score

from MBC import MBCParams, standardize_then_mbc

SEEDS = [7, 11, 23]
N_VALUES = [200, 500, 1000, 2000, 5000]


def make_dataset(name: str, n: int, seed: int):
    if name == "blobs_2D":
        return make_blobs(n_samples=n, centers=4, n_features=2,
                          cluster_std=0.9, random_state=seed)
    if name == "moons":
        return make_moons(n_samples=n, noise=0.10, random_state=seed)
    if name == "blobs_50D_easy":
        return make_blobs(n_samples=n, centers=6, n_features=50,
                          cluster_std=1.0, random_state=seed)
    raise ValueError(name)


def regime_flag(rho_hat: float, C_low: float, C_upper: float) -> str:
    if rho_hat is None or rho_hat < 0:
        return "non_separable"
    if rho_hat > C_upper:
        return "separable"
    if rho_hat < C_low:
        return "non_separable"
    return "transitional"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default="results/sampling")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--n_values", type=int, nargs="+", default=N_VALUES)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "blobs_2D": 4,
        "moons": 2,
        "blobs_50D_easy": 6,
    }
    rows = []
    for name, K_true in datasets.items():
        for n in args.n_values:
            for seed in args.seeds:
                X, y = make_dataset(name, n, seed)
                res = standardize_then_mbc(X, MBCParams(), seed=seed)
                ari = adjusted_rand_score(y, res.labels)
                rows.append(dict(
                    dataset=name, K_true=K_true, n=int(X.shape[0]),
                    seed=seed,
                    K=int(res.n_clusters),
                    bracket_low=int(res.bracket[0]),
                    bracket_high=int(res.bracket[1]),
                    k_star=int(res.k_star),
                    rho_hat=float(res.rho_hat),
                    C_low=float(res.C_lower),
                    C_upper=float(res.C_upper),
                    regime=regime_flag(res.rho_hat, res.C_lower, res.C_upper),
                    ARI=float(ari),
                ))
                r = rows[-1]
                print(f"  {name:<16} n={n:<5} seed={seed}  "
                      f"bracket=[{r['bracket_low']},{r['bracket_high']}] "
                      f"K={r['K']} k*={r['k_star']} rho={r['rho_hat']:.2f} "
                      f"reg={r['regime']:<13} ARI={r['ARI']:.2f}")
    df = pd.DataFrame(rows)
    raw_path = out_dir / "sampling_sweep.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nWrote {raw_path} ({len(df)} rows)")

    # Summary: median per (dataset, n)
    agg = df.groupby(["dataset", "K_true", "n"]).agg(
        K_prac=("K", "median"),
        bracket_low=("bracket_low", "median"),
        bracket_high=("bracket_high", "median"),
        k_star=("k_star", "median"),
        rho_hat=("rho_hat", "median"),
        regime=("regime", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
        ARI=("ARI", "mean"),
    ).reset_index()
    summary_path = out_dir / "sampling_sweep_summary.csv"
    agg.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")
    print("\n", agg.to_string(index=False))


if __name__ == "__main__":
    main()
