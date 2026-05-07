#!/usr/bin/env python3
"""No-pruning ablation: rerun the contamination case with
use_density_pruning=False to confirm the bracket collapses earlier
without the active-set restriction.
"""
from __future__ import annotations
import csv
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from MBC import MBCParams, mbc_cluster  # noqa: E402
from scripts.run_A_sweep import gen_blobs_bg, gen_moons_noisy  # noqa: E402

SEEDS = [7, 11, 23]
N = 2000
DATASETS = [
    ("blobs_2D_bg10", 4, gen_blobs_bg),
    ("moons_n0.15",   2, gen_moons_noisy),
]


def run_one(name, K_true, X, y, use_pruning, seed):
    Xs = StandardScaler().fit_transform(X)
    p = MBCParams(delta=0.05, A_coef=1.0, candidate_mode="mutual",
                  use_density_pruning=use_pruning)
    t0 = time.time()
    res = mbc_cluster(Xs, p, seed=seed)
    runtime = time.time() - t0
    mask = y >= 0
    ari = float("nan")
    if mask.sum() > 1:
        try:
            ari = adjusted_rand_score(y[mask], res.labels[mask])
        except Exception:
            ari = float("nan")
    info = res.info or {}
    K_low_b, K_high_b = res.bracket
    return {
        "dataset": name,
        "K_true": K_true,
        "use_pruning": use_pruning,
        "seed": seed,
        "k_star": int(res.k_star),
        "K_low": int(K_low_b),
        "K_high": int(K_high_b),
        "K_prac": int(res.n_clusters),
        "ARI": ari,
        "regime": info.get("regime", ""),
        "rho_hat": float(res.rho_hat),
        "n_active": int(info.get("n_active", -1)),
        "runtime": runtime,
    }


def main():
    out_dir = ROOT / "results" / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pruning_ablation.csv"
    rows = []
    for name, K_true, gen in DATASETS:
        for seed in SEEDS:
            X, y = gen(N, seed)
            for use in [True, False]:
                row = run_one(name, K_true, X, y, use, seed)
                rows.append(row)
                print(f"{name:18s} prune={use!s:5s} seed={seed} "
                      f"K=[{row['K_low']},{row['K_high']}] "
                      f"K^={row['K_prac']} rho={row['rho_hat']:.2f} "
                      f"reg={row['regime']:12s} ARI={row['ARI']:.3f} "
                      f"n_active={row['n_active']}")
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
