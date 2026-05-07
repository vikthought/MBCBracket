#!/usr/bin/env python3
"""
run_real.py — real-data benchmark for MBC.

Datasets and where they come from:
  - MNIST_pca50, FashionMNIST_pca50, 20NG_svd100, Digits_pca50  -> data/*.npz
  - CIFAR10                                                       -> data/cifar-10-batches-py/
  - Iris, Wine, BreastCancer                                      -> sklearn (small)
  - Olivetti faces                                                -> sklearn (fetch)
  - PenDigits                                                     -> openml 'pendigits' (fetch)
  - LetterRecognition                                             -> openml 'letter' (fetch)

Each loader caches its processed result under data/ when feasible. Loaders
return None when their dependency (e.g. internet, torchvision, openml) is
unavailable; the runner just skips those datasets.

Usage:
    python run_real.py
    python run_real.py --datasets MNIST Iris --max_n 4000
    python run_real.py --dtm off --seeds 1 2
    python run_real.py --bracket_eval mid_max
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from mbc_runner import (
    DatasetSpec,
    RunWriter,
    add_common_args,
    dtm_modes,
    filter_specs,
    pca_if_needed,
    run_dataset,
    zscore,
)

DATA_DIR = Path("data")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_npz(path: Path):
    if not path.exists():
        return None
    try:
        d = np.load(path)
        return d["X"], d["y"]
    except Exception as e:
        print(f"  [{path.name}] load failed: {e}"); return None


def _subsample(X: np.ndarray, y: np.ndarray, max_n: int, seed: int):
    if len(X) <= max_n:
        return X, y
    idx = np.random.default_rng(seed).choice(len(X), size=max_n, replace=False)
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# Loader factory: returns a callable suitable for DatasetSpec.loader.
# ---------------------------------------------------------------------------

def _mk_npz_loader(npz_name: str, max_n_default: int = 8000):
    def loader(seed: int, _npz=npz_name, _max=max_n_default):
        cached = _load_npz(DATA_DIR / _npz)
        if cached is None:
            return None
        return _subsample(*cached, _max, seed)
    return loader


def _mk_max_n_loader(loader_fn, max_n: int):
    def wrapped(seed: int):
        data = loader_fn(seed)
        if data is None:
            return None
        return _subsample(*data, max_n, seed)
    return wrapped


def _load_mnist_with_fetch(seed: int):
    cached = _load_npz(DATA_DIR / "MNIST_pca50_n8000.npz")
    if cached is not None:
        return cached
    try:
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml("mnist_784", version=1, as_frame=False,
                            return_X_y=True, cache=True)
        X = X.astype(np.float32); y = y.astype(int)
        X, y = _subsample(X, y, 8000, seed)
        Xp = pca_if_needed(X, target_dim=50, seed=seed)
        np.savez_compressed(DATA_DIR / "MNIST_pca50_n8000.npz", X=Xp, y=y)
        return Xp, y
    except Exception as e:
        print(f"  [MNIST] fetch failed: {e}"); return None


def _load_cifar10(seed: int):
    cifar_dir = DATA_DIR / "cifar-10-batches-py"
    if not cifar_dir.exists():
        return None
    try:
        Xs, ys = [], []
        for i in range(1, 6):
            with open(cifar_dir / f"data_batch_{i}", "rb") as f:
                d = pickle.load(f, encoding="latin1")
            Xs.append(d["data"].astype(np.float32))
            ys.append(np.array(d["labels"], dtype=int))
        X = np.vstack(Xs); y = np.concatenate(ys)
        X, y = _subsample(X, y, 8000, seed)
        Xp = pca_if_needed(X, target_dim=50, seed=seed)
        return Xp, y
    except Exception as e:
        print(f"  [CIFAR10] load failed: {e}"); return None


def _load_iris(seed: int):
    try:
        from sklearn.datasets import load_iris
        d = load_iris(); return zscore(d.data.astype(np.float32)), d.target.astype(int)
    except Exception:
        return None


def _load_wine(seed: int):
    try:
        from sklearn.datasets import load_wine
        d = load_wine(); return zscore(d.data.astype(np.float32)), d.target.astype(int)
    except Exception:
        return None


def _load_breast_cancer(seed: int):
    try:
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer(); return zscore(d.data.astype(np.float32)), d.target.astype(int)
    except Exception:
        return None


def _load_olivetti(seed: int):
    cached = _load_npz(DATA_DIR / "Olivetti_pca50.npz")
    if cached is not None:
        return cached
    try:
        from sklearn.datasets import fetch_olivetti_faces
        d = fetch_olivetti_faces(random_state=seed)
        X = d.data.astype(np.float32); y = d.target.astype(int)
        Xp = pca_if_needed(X, target_dim=50, seed=seed)
        np.savez_compressed(DATA_DIR / "Olivetti_pca50.npz", X=Xp, y=y)
        return Xp, y
    except Exception as e:
        print(f"  [Olivetti] fetch failed: {e}"); return None


def _load_pendigits(seed: int):
    cached = _load_npz(DATA_DIR / "Pendigits.npz")
    if cached is not None:
        return cached
    try:
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml("pendigits", as_frame=False, return_X_y=True,
                            cache=True)
        X = X.astype(np.float32); y = y.astype(int)
        Xp = zscore(X)
        np.savez_compressed(DATA_DIR / "Pendigits.npz", X=Xp, y=y)
        return Xp, y
    except Exception as e:
        print(f"  [Pendigits] fetch failed: {e}"); return None


def _load_letter(seed: int):
    cached = _load_npz(DATA_DIR / "Letter.npz")
    if cached is not None:
        return cached
    try:
        from sklearn.datasets import fetch_openml
        X, y_str = fetch_openml("letter", version=1, as_frame=False,
                                return_X_y=True, cache=True)
        X = X.astype(np.float32)
        # y is letters A-Z
        if y_str.dtype.kind in "OUS":
            uniq = sorted(set(y_str.tolist()))
            mp = {v: i for i, v in enumerate(uniq)}
            y = np.array([mp[v] for v in y_str], dtype=int)
        else:
            y = y_str.astype(int)
        Xp = zscore(X)
        np.savez_compressed(DATA_DIR / "Letter.npz", X=Xp, y=y)
        return Xp, y
    except Exception as e:
        print(f"  [Letter] fetch failed: {e}"); return None


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------

def real_specs(max_n: int) -> list:
    return [
        DatasetSpec(name="MNIST", family="image", K_true=10,
                    loader=_mk_max_n_loader(_load_mnist_with_fetch, max_n)),
        DatasetSpec(name="FashionMNIST", family="image", K_true=10,
                    loader=_mk_max_n_loader(
                        _mk_npz_loader("FashionMNIST_pca50_n8000.npz", 8000),
                        max_n)),
        DatasetSpec(name="CIFAR10", family="image", K_true=10,
                    loader=_mk_max_n_loader(_load_cifar10, max_n)),
        DatasetSpec(name="Olivetti", family="image", K_true=40,
                    loader=_mk_max_n_loader(_load_olivetti, max_n)),
        DatasetSpec(name="20NG", family="text", K_true=20,
                    loader=_mk_max_n_loader(
                        _mk_npz_loader("20NG_svd100_n8000.npz", 8000),
                        max_n)),
        DatasetSpec(name="Digits_pca50", family="tabular", K_true=10,
                    loader=_mk_max_n_loader(
                        _mk_npz_loader("Digits_pca50.npz", 8000), max_n)),
        DatasetSpec(name="Pendigits", family="tabular", K_true=10,
                    loader=_mk_max_n_loader(_load_pendigits, max_n)),
        DatasetSpec(name="Letter", family="tabular", K_true=26,
                    loader=_mk_max_n_loader(_load_letter, max_n)),
        DatasetSpec(name="Iris", family="small_tabular", K_true=3,
                    loader=_load_iris),
        DatasetSpec(name="Wine", family="small_tabular", K_true=3,
                    loader=_load_wine),
        DatasetSpec(name="BreastCancer", family="small_tabular", K_true=2,
                    loader=_load_breast_cancer),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--max_n", type=int, default=8000,
                    help="Subsample real datasets to at most this many points.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir or "results/real")
    writer = RunWriter(out_dir=out_dir, suite="real")

    specs = filter_specs(real_specs(args.max_n), args.datasets, args.families)
    if not specs:
        print("No datasets selected after filtering."); return

    dtm_choices = dtm_modes(args.dtm)
    print(f"Real suite: {len(specs)} datasets x {len(args.seeds)} seeds, "
          f"DTM={args.dtm}, max_n={args.max_n}, "
          f"bracket_eval={args.bracket_eval}")

    rows = []
    for seed in args.seeds:
        for spec in specs:
            data = spec.loader(seed)
            if data is None:
                print(f"\n=== {spec.name} seed={seed}: not available, skipping ===")
                continue
            X, y = data
            print(f"\n=== {spec.name} [{spec.family}] seed={seed} "
                  f"(n={len(X)}, D={X.shape[1]}, K_true={spec.K_true}) ===")
            rows.extend(run_dataset(
                spec=spec, X=X, y=y, seed=seed,
                dtm_choices=dtm_choices,
                figs_dir=writer.figs_dir,
                bracket_eval=args.bracket_eval,
                skip_baselines=args.no_baselines,
            ))

    writer.write(rows)


if __name__ == "__main__":
    main()
