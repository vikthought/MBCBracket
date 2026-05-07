#!/usr/bin/env python3
"""Smoke + regression test for MBC.

Two layers:

* **Smoke**: shape / API sanity. Bracket monotone, bracket_labels dict has
  K-many labels per K, get_labels_for_K resolves.

* **Regression**: pin bracket / K_practical / regime on three reference
  datasets at seed 7 (the seed used for results/synth/synth_raw.csv).
  Catches accidental
  drift in the bracket numbers. Tolerances are generous — the bracket is
  asserted exactly, ARI within 0.05 — so legitimate small numerical
  changes are fine but a regression that shifts the bracket trips the
  test.

Run with::

    python test_mbc_smoke.py
"""
from __future__ import annotations

import sys

import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score

from MBC import (MBCParams, get_labels_for_K, mbc_cluster, standardize_then_mbc,
                 summarize)


# =============================================================================
# Smoke (shape sanity)
# =============================================================================

def _toy_three_clusters(seed: int = 0, D: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = np.zeros((3, D))
    centers[1, 0] = 5.0
    centers[2, 1] = 5.0
    return np.vstack([rng.normal(c, 0.3, (200, D)) for c in centers])


def _check_smoke(use_dtm: bool, seed: int) -> None:
    X = _toy_three_clusters(seed)
    p = MBCParams(); p.use_dtm_rescue = use_dtm
    res = standardize_then_mbc(X, p, seed=seed)
    print(summarize(res))

    n = X.shape[0]
    assert res.labels.shape == (n,), "labels shape mismatch"
    assert res.bracket[0] <= res.bracket[1], "bracket not monotone"
    assert res.k_low <= res.k_high, "k bracket not monotone"

    assert isinstance(res.bracket_labels, dict)
    assert len(res.bracket_labels) >= 1, "bracket_labels empty"
    for K, labs in res.bracket_labels.items():
        assert labs.shape == (n,), f"bracket_labels[{K}] wrong shape"
        n_unique = int(np.unique(labs).size)
        assert n_unique == K, f"bracket_labels[{K}] has {n_unique} unique"

    for K in res.bracket_labels:
        assert K in res.bracket_k_for_K, f"k for K={K} missing"

    Ks = sorted(res.bracket_labels.keys())
    mid_K = Ks[len(Ks) // 2]
    labs_mid = get_labels_for_K(res, mid_K)
    assert labs_mid is not None and labs_mid.shape == (n,)
    print(f"[smoke ok] use_dtm={use_dtm} seed={seed} "
          f"bracket={res.bracket} bracket_Ks={Ks}\n")


# =============================================================================
# Regression (numeric pins)
#
# Each entry: (dataset name, generator, expected bracket, expected K_prac,
# minimum ARI). Generators must be deterministic at seed 7. The pinned
# numbers come from the seed-7 results in results/synth/synth_raw.csv that
# back the catalog table in PAPER/main.tex.
# =============================================================================

REGRESSION_CASES = [
    {
        "name": "blobs_2D_clean",
        "gen": lambda: make_blobs(n_samples=2000, centers=4, n_features=2,
                                  cluster_std=0.9, random_state=7),
        "bracket": (4, 4),
        "K_prac": 4,
        "ARI_min": 0.95,
    },
    {
        "name": "moons_n0.05",
        "gen": lambda: make_moons(n_samples=2000, noise=0.05, random_state=7),
        "bracket": (2, 2),
        "K_prac": 2,
        "ARI_min": 0.95,
    },
    {
        "name": "blobs_50D_easy",
        "gen": lambda: make_blobs(n_samples=2000, centers=6, n_features=50,
                                  cluster_std=1.0, random_state=7),
        "bracket": (6, 6),
        "K_prac": 6,
        "ARI_min": 0.95,
    },
]


def _check_regression(case: dict) -> None:
    X, y = case["gen"]()
    res = standardize_then_mbc(X, MBCParams(), seed=7)
    ari = adjusted_rand_score(y, res.labels)

    actual_bracket = (int(res.bracket[0]), int(res.bracket[1]))
    actual_Kprac = int(res.n_clusters)

    print(f"  {case['name']}: bracket={actual_bracket}  K_prac={actual_Kprac}  "
          f"ARI={ari:.3f}")

    assert actual_bracket == case["bracket"], (
        f"{case['name']}: bracket {actual_bracket} != expected {case['bracket']}"
    )
    assert actual_Kprac == case["K_prac"], (
        f"{case['name']}: K_prac {actual_Kprac} != expected {case['K_prac']}"
    )
    assert ari >= case["ARI_min"], (
        f"{case['name']}: ARI {ari:.3f} below floor {case['ARI_min']}"
    )


# =============================================================================
# Entry point
# =============================================================================

def main() -> int:
    print("=" * 60)
    print("Smoke layer")
    print("=" * 60)
    _check_smoke(use_dtm=True,  seed=0)
    _check_smoke(use_dtm=False, seed=1)

    print("=" * 60)
    print("Regression layer (seed-7 pins from synth_raw.csv)")
    print("=" * 60)
    for case in REGRESSION_CASES:
        _check_regression(case)

    print("\nsmoke + regression test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
