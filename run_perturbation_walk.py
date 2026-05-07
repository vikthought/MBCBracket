#!/usr/bin/env python3
"""
run_perturbation_walk.py — backs `tab:perturbation-walk` (app:perturbation-walk in PAPER/main.tex).

Three controlled axes on a clean 4-blob 2D base at n=1500:
  - contamination eta in {0, 0.05, 0.10, 0.20}
  - cluster spread sigma multiplier in {1, 2, 4, 8}
  - centroid distance Delta in {10, 5, 4, 2}

Each axis holds the other two at the easy baseline. Three seeds per cell
(7, 11, 23). Outputs: results/perturbations/perturbation_walk.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from MBC import MBCParams, standardize_then_mbc


N_BASE = 1500
SIGMA_BASE = 1.0
DELTA_BASE = 10.0
SEEDS = [7, 11, 23]


def make_base_blobs(seed: int, n: int = N_BASE,
                    sigma_mult: float = 1.0,
                    delta: float = DELTA_BASE):
    """Four 2D blobs on a square at +/- delta/2."""
    half = delta / 2.0
    centers = np.array([[-half, -half], [-half, half],
                        [half, -half], [half, half]])
    X, y = make_blobs(n_samples=n, centers=centers,
                      cluster_std=SIGMA_BASE * sigma_mult,
                      random_state=seed)
    return X, y


def add_uniform_bg(X, y, eta: float, seed: int):
    if eta <= 0:
        return X, y
    rng = np.random.default_rng(seed)
    lo, hi = X.min(0), X.max(0)
    pad = 0.1 * (hi - lo + 1e-9)
    n_noise = max(1, int(eta * len(X) / max(1e-9, 1.0 - eta)))
    bg = rng.uniform(lo - pad, hi + pad, size=(n_noise, X.shape[1]))
    Xa = np.vstack([X, bg])
    ya = np.concatenate([y, np.full(n_noise, -1, dtype=int)])
    perm = rng.permutation(len(Xa))
    return Xa[perm], ya[perm]


def regime_flag(rho_hat: float, C_low: float, C_upper: float) -> str:
    if rho_hat is None or rho_hat < 0:
        return "non_separable"
    if rho_hat > C_upper:
        return "separable"
    if rho_hat < C_low:
        return "non_separable"
    return "transitional"


def evaluate(X: np.ndarray, y: np.ndarray, seed: int):
    res = standardize_then_mbc(X, MBCParams(), seed=seed)
    valid = y >= 0
    if valid.sum() == 0:
        ari = nmi = float("nan")
    else:
        ari = adjusted_rand_score(y[valid], res.labels[valid])
        nmi = normalized_mutual_info_score(y[valid], res.labels[valid])
    return dict(
        K=int(res.n_clusters),
        bracket_low=int(res.bracket[0]),
        bracket_high=int(res.bracket[1]),
        rho_hat=float(res.rho_hat),
        C_low=float(res.C_lower),
        C_upper=float(res.C_upper),
        regime=regime_flag(res.rho_hat, res.C_lower, res.C_upper),
        ARI=float(ari),
        NMI=float(nmi),
        k_star=int(res.k_star),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default="results/perturbations")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    # Axis 1: contamination
    for eta in [0.0, 0.05, 0.10, 0.20]:
        for seed in args.seeds:
            X, y = make_base_blobs(seed)
            X, y = add_uniform_bg(X, y, eta, seed)
            r = evaluate(X, y, seed)
            r.update(axis="contamination", setting=f"eta={eta}",
                     eta=eta, sigma_mult=1.0, delta=DELTA_BASE,
                     n=int(X.shape[0]), K_true=4, seed=seed)
            rows.append(r)
            print(f"  eta={eta} seed={seed}  bracket=[{r['bracket_low']},{r['bracket_high']}] "
                  f"K={r['K']} rho={r['rho_hat']:.2f} regime={r['regime']} ARI={r['ARI']:.2f}")

    # Axis 2: cluster spread
    for mult in [1.0, 2.0, 4.0, 8.0]:
        for seed in args.seeds:
            X, y = make_base_blobs(seed, sigma_mult=mult)
            r = evaluate(X, y, seed)
            r.update(axis="cluster_spread", setting=f"sigma_mult={mult}",
                     eta=0.0, sigma_mult=mult, delta=DELTA_BASE,
                     n=int(X.shape[0]), K_true=4, seed=seed)
            rows.append(r)
            print(f"  sigma_mult={mult} seed={seed}  bracket=[{r['bracket_low']},{r['bracket_high']}] "
                  f"K={r['K']} rho={r['rho_hat']:.2f} regime={r['regime']} ARI={r['ARI']:.2f}")

    # Axis 3: centroid distance
    for delta in [10.0, 5.0, 4.0, 2.0]:
        K_true = 4 if delta > 0 else 1
        for seed in args.seeds:
            X, y = make_base_blobs(seed, delta=delta)
            r = evaluate(X, y, seed)
            r.update(axis="centroid_distance", setting=f"Delta={delta}",
                     eta=0.0, sigma_mult=1.0, delta=delta,
                     n=int(X.shape[0]), K_true=K_true, seed=seed)
            rows.append(r)
            print(f"  Delta={delta} seed={seed}  bracket=[{r['bracket_low']},{r['bracket_high']}] "
                  f"K={r['K']} rho={r['rho_hat']:.2f} regime={r['regime']} ARI={r['ARI']:.2f}")

    df = pd.DataFrame(rows)
    raw_path = out_dir / "perturbation_walk.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nWrote {raw_path} ({len(df)} rows)")

    # Summary: median across seeds per cell
    agg = df.groupby(["axis", "setting", "K_true"]).agg(
        bracket_low=("bracket_low", "median"),
        bracket_high=("bracket_high", "median"),
        K_prac=("K", "median"),
        rho_hat=("rho_hat", "median"),
        regime=("regime", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
        ARI=("ARI", "mean"),
    ).reset_index()
    summary_path = out_dir / "perturbation_walk_summary.csv"
    agg.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")
    print("\n", agg.to_string(index=False))


if __name__ == "__main__":
    main()
