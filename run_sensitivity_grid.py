#!/usr/bin/env python3
"""
run_sensitivity_grid.py — backs `tab:sensitivity` (app:sensitivity in PAPER/main.tex).

A 4x4 grid in (delta, alpha) on three datasets:
  - 50D blobs easy (separable; K=6)
  - Retina labeled (non-separable transitional; K=7)
  - two moons noise=0.10 (non-separable; K=2)

Each cell uses three seeds (7, 11, 23). Output:
  results/sensitivity/sensitivity_grid.csv

Note: the Retina_labeled dataset is loaded from data/retina_diffmap.npy and
the diffusion-map embedding is fed directly to mbc_cluster (no z-score) to
preserve geodesic geometry, matching the run_neuro.py pipeline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score

from MBC import MBCParams, mbc_cluster, standardize_then_mbc

SEEDS = [7, 11, 23]
DELTAS = [0.01, 0.05, 0.10, 0.20]
ALPHAS = [1.0, 1.25, 1.5, 2.0]


def load_retina_labeled():
    Psi = np.load(Path("data") / "retina_diffmap.npy")
    rgc = np.load(Path("data") / "rgc_types.npy", allow_pickle=True)
    types = np.array([str(t) for t in rgc])
    type_to_int = {t: i for i, t in enumerate(sorted(np.unique(types)))}
    color = np.array([type_to_int[t] for t in types], dtype=int)
    mask = color > 0
    return Psi[mask], color[mask]


def make_dataset(name: str, seed: int):
    if name == "blobs_50D_easy":
        return make_blobs(n_samples=2000, centers=6, n_features=50,
                          cluster_std=1.0, random_state=seed), "synth"
    if name == "moons_n0.10":
        return make_moons(n_samples=2000, noise=0.10, random_state=seed), "synth"
    if name == "Retina_labeled":
        return load_retina_labeled(), "retina"
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
    ap.add_argument("--out_dir", default="results/sensitivity")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "blobs_50D_easy": 6,
        "Retina_labeled": 7,
        "moons_n0.10": 2,
    }
    rows = []
    for ds_name, K_true in datasets.items():
        for delta in DELTAS:
            for alpha in ALPHAS:
                for seed in args.seeds:
                    (X, y), kind = make_dataset(ds_name, seed)
                    p = MBCParams()
                    p.delta = float(delta)
                    p.min_h_gate_alpha = float(alpha)
                    p.noise_factor = float(alpha)  # alpha_q tied to alpha
                    if kind == "retina":
                        # Diffusion-map embedding: skip z-score
                        res = mbc_cluster(X, p, seed=seed)
                    else:
                        res = standardize_then_mbc(X, p, seed=seed)
                    if y is None:
                        ari = float("nan")
                    else:
                        ari = adjusted_rand_score(y, res.labels)
                    rows.append(dict(
                        dataset=ds_name, K_true=K_true,
                        delta=delta, alpha=alpha, seed=seed,
                        K=int(res.n_clusters),
                        bracket_low=int(res.bracket[0]),
                        bracket_high=int(res.bracket[1]),
                        k_star=int(res.k_star),
                        rho_hat=float(res.rho_hat),
                        regime=regime_flag(res.rho_hat, res.C_lower, res.C_upper),
                        ARI=float(ari),
                    ))
                    r = rows[-1]
                    print(f"  {ds_name:<16} d={delta:<5} a={alpha:<5} seed={seed}  "
                          f"bracket=[{r['bracket_low']},{r['bracket_high']}] "
                          f"K={r['K']} rho={r['rho_hat']:.2f} "
                          f"reg={r['regime']:<13} ARI={r['ARI']:.2f}")
    df = pd.DataFrame(rows)
    raw_path = out_dir / "sensitivity_grid.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nWrote {raw_path} ({len(df)} rows)")

    # Summary: per dataset, min/median/max of bracket / K_prac / ARI across grid x seeds
    print("\n=== summary per dataset ===")
    for ds in datasets:
        sub = df[df['dataset']==ds]
        b_lo_min = sub['bracket_low'].min(); b_lo_med = sub['bracket_low'].median(); b_lo_max = sub['bracket_low'].max()
        b_hi_min = sub['bracket_high'].min(); b_hi_med = sub['bracket_high'].median(); b_hi_max = sub['bracket_high'].max()
        k_min = sub['K'].min(); k_med = sub['K'].median(); k_max = sub['K'].max()
        ari_min = sub['ARI'].min(); ari_med = sub['ARI'].median(); ari_max = sub['ARI'].max()
        print(f"  {ds}: bracket = [{b_lo_min:.0f},{b_hi_min:.0f}] / "
              f"[{b_lo_med:.0f},{b_hi_med:.0f}] / [{b_lo_max:.0f},{b_hi_max:.0f}]  "
              f"K = {k_min:.0f}/{k_med:.0f}/{k_max:.0f}  "
              f"ARI = {ari_min:.3f}/{ari_med:.3f}/{ari_max:.3f}")
        # Headline cell: delta=0.05, alpha=1.5
        head = sub[(sub['delta']==0.05) & (sub['alpha']==1.5)]
        if not head.empty:
            print(f"    headline (delta=0.05, alpha=1.5): "
                  f"bracket=[{head['bracket_low'].median():.0f},{head['bracket_high'].median():.0f}] "
                  f"K={head['K'].median():.0f} ARI={head['ARI'].mean():.3f}")


if __name__ == "__main__":
    main()
