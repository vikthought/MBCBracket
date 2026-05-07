#!/usr/bin/env python3
"""Compute (per-dataset) mass-coverage brackets at gamma in {0.90, 0.95, 0.98}
using the bracket_sweep already produced by mbc_cluster.

For each dataset on synth/real/neuro suites we run MBC once (matching the
preprocessing in run_synth.py / run_real.py / run_neuro.py) and then call
the existing _mass_K_at_step + _mass_bracket_from_zone helpers at three
gamma values. All three brackets use the persistent variant (run_min=2).

Output: results/gamma_sensitivity.csv  with columns
  suite, dataset, K_true, headline_low, headline_high,
  mass090_low, mass090_high, mass095_low, mass095_high,
  mass098_low, mass098_high, regime, ARI, NMI
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from MBC import (
    MBCParams,
    mbc_cluster,
    _mass_K_at_step,
    _mass_bracket_from_zone,
)
from run_synth import synth_specs
from run_neuro import (
    _load_retina_full,
    _load_retina_labeled,
    _load_retina_off_brisk,
    _load_retina_non_off_brisk,
    _load_v1,
)

DATA = ROOT / "data"
OUT = ROOT / "results" / "gamma_sensitivity.csv"

SEED = 7
GAMMAS = (0.90, 0.95, 0.98)


def _brackets_at_gamma(res, gamma: float
                       ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Return (per-step, persistent run>=2) mass-gamma brackets from sweep."""
    first = res.bracket_sweep[0]
    active = first["labels"] >= 0
    active_idx = np.where(active)[0]
    in_zone = [s for s in res.bracket_sweep
               if res.k_low <= s["k"] <= res.k_high]
    if not in_zone:
        in_zone = res.bracket_sweep[:]
    mass_K = [_mass_K_at_step(s["labels"][active_idx], gamma) for s in in_zone]
    step, persistent = _mass_bracket_from_zone(mass_K, 2)
    return (int(step[0]), int(step[1])), (int(persistent[0]), int(persistent[1]))


def _eval(y_true, labels):
    if y_true is None or len(y_true) == 0:
        return float("nan"), float("nan")
    return (adjusted_rand_score(y_true, labels),
            normalized_mutual_info_score(y_true, labels))


def _record(rows: List[dict], suite: str, name: str, X: np.ndarray,
            y, K_true, *, standardize: bool, pca_mode: str) -> None:
    p = MBCParams()
    t0 = time.time()
    res = mbc_cluster(X, p, seed=SEED,
                      standardize=standardize, pca_mode=pca_mode)
    dt = time.time() - t0
    ari, nmi = _eval(y, res.labels)
    row = dict(
        suite=suite,
        dataset=name,
        n=int(X.shape[0]),
        D=int(X.shape[1]),
        K_true=K_true,
        headline_low=int(res.bracket[0]),
        headline_high=int(res.bracket[1]),
        regime=res.info.get("regime"),
        ARI=ari,
        NMI=nmi,
        runtime_s=round(dt, 2),
    )
    for g in GAMMAS:
        step, persistent = _brackets_at_gamma(res, g)
        tag = f"mass{int(g*100):03d}"
        row[f"{tag}_step_low"] = step[0]
        row[f"{tag}_step_high"] = step[1]
        row[f"{tag}_pers_low"] = persistent[0]
        row[f"{tag}_pers_high"] = persistent[1]
    rows.append(row)
    print(f"  {suite}/{name}: head=[{row['headline_low']},{row['headline_high']}]"
          f" m95=[{row['mass095_step_low']},{row['mass095_step_high']}]"
          f" m95_pers=[{row['mass095_pers_low']},{row['mass095_pers_high']}]"
          f"  ARI={ari:.3f} ({dt:.1f}s)", flush=True)


def run_synth(rows):
    for spec in synth_specs(n=2000):
        d = spec.loader(SEED)
        if d is None:
            continue
        X, y = d
        try:
            _record(rows, "synth", spec.name, X, y, spec.K_true,
                    standardize=spec.standardize_for_mbc,
                    pca_mode=("project_90" if spec.standardize_for_mbc
                              else "none"))
        except Exception as e:
            print(f"    synth/{spec.name} failed: {e}")


def run_real(rows):
    real_files = [
        ("Iris", "Iris", 3),
        ("Wine", "Wine", 3),
        ("BreastCancer", "BreastCancer", 2),
        ("Olivetti", "Olivetti_pca50", 40),
        ("Digits_pca50", "Digits_pca50", 10),
        ("Pendigits", "Pendigits", 10),
        ("Letter", "Letter", 26),
        ("MNIST", "MNIST_pca50_n8000", 10),
        ("FashionMNIST", "FashionMNIST_pca50_n8000", 10),
    ]
    for name, fname, K_true in real_files:
        from sklearn.datasets import (
            load_iris, load_wine, load_breast_cancer, load_digits,
            fetch_olivetti_faces,
        )
        path = DATA / f"{fname}.npz"
        X = y = None
        if path.exists():
            try:
                z = np.load(path, allow_pickle=True)
                X, y = z["X"], z["y"]
            except Exception as e:
                pass
        if X is None:
            try:
                if name == "Iris":
                    d = load_iris(); X, y = d.data, d.target
                elif name == "Wine":
                    d = load_wine(); X, y = d.data, d.target
                elif name == "BreastCancer":
                    d = load_breast_cancer(); X, y = d.data, d.target
                elif name == "Olivetti":
                    d = fetch_olivetti_faces(); X, y = d.data, d.target
                    from sklearn.decomposition import PCA
                    X = PCA(n_components=50, random_state=0).fit_transform(X)
                elif name == "Digits_pca50":
                    d = load_digits(); X, y = d.data, d.target
                    from sklearn.decomposition import PCA
                    X = PCA(n_components=50, random_state=0).fit_transform(X)
                else:
                    print(f"  real/{name}: missing {path} and no fallback")
                    continue
            except Exception as e:
                print(f"  real/{name}: load failed: {e}")
                continue
        try:
            _record(rows, "real", name, X, y, K_true,
                    standardize=True, pca_mode="project_90")
        except Exception as e:
            print(f"    real/{name} failed: {e}")


def run_neuro(rows):
    loaders = [
        ("Retina_full", _load_retina_full, None),
        ("Retina_labeled", _load_retina_labeled, 7),
        ("Retina_off_brisk", _load_retina_off_brisk, 2),
        ("Retina_non_off_brisk", _load_retina_non_off_brisk, None),
        ("V1", _load_v1, None),
    ]
    for name, fn, K_true in loaders:
        d = fn(SEED)
        if d is None:
            continue
        X, y = d
        try:
            _record(rows, "neuro", name, X, y, K_true,
                    standardize=False, pca_mode="none")
        except Exception as e:
            print(f"    neuro/{name} failed: {e}")


def main():
    t0 = time.time()
    rows = []
    print("=== Synth ===", flush=True);    run_synth(rows)
    print("=== Real ===", flush=True);     run_real(rows)
    print("=== Neuro ===", flush=True);    run_neuro(rows)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nTotal {len(rows)} rows; saved to {OUT}")
    print(f"Wall: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
