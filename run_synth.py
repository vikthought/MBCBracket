#!/usr/bin/env python3
"""
run_synth.py — synthetic-data benchmark for MBC.

Suite is organized by `family`:
  classic         — moons, circles, blobs (low-D)
  noise_sweep     — moons & circles at multiple noise levels
  bg_noise        — clusters with uniform background-noise contamination
                    (varying fractions: 5/10/20%)
  varied          — anisotropic, varied per-cluster std, swiss-roll
  high_D          — Gaussian blobs at D in {50, 100, 200} with varying
                    separability and anisotropy
  hierarchical    — clusters of clusters
  adversarial     — touching gaussians, uneven blobs, helix-plane-sphere
  imbalanced      — biased cluster-size mixtures (90/10, 80/15/5, etc.)
                    Real data is rarely balanced; this probes whether
                    the bracket and K-rule survive class imbalance.

Reference: dataset selection inspired by the HDBSCAN paper, density-peaks
(Rodriguez & Laio 2014) benchmarks, and standard clustering surveys (S-/A-set
families, R15/D31, etc., approximated with synthetic generators).

Usage:
    python run_synth.py
    python run_synth.py --dtm off --families noise_sweep
    python run_synth.py --datasets moons --seeds 1 2 3
    python run_synth.py --n 1500 --bracket_eval mid_max
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons, make_swiss_roll

from mbc_runner import (
    DatasetSpec,
    RunWriter,
    add_common_args,
    dtm_modes,
    filter_specs,
    run_dataset,
)


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def _aniso_blobs(n: int, seed: int, centers: int = 4, std: float = 1.0,
                 D: int = 2):
    X, y = make_blobs(n_samples=n, centers=centers, n_features=D,
                      cluster_std=std, random_state=seed)
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(D, D))
    U, _, Vt = np.linalg.svd(A, full_matrices=False)
    S = np.diag(np.linspace(2.5, 0.4, D))
    return X @ (U @ S @ Vt).T, y


def _add_uniform_bg_noise(X: np.ndarray, y: np.ndarray, frac: float,
                          seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if frac <= 0:
        return X, y
    rng = np.random.default_rng(seed)
    lo, hi = X.min(axis=0), X.max(axis=0)
    pad = 0.1 * (hi - lo + 1e-9)
    n_noise = max(1, int(frac * len(X) / max(1e-9, 1.0 - frac)))
    noise = rng.uniform(lo - pad, hi + pad, size=(n_noise, X.shape[1]))
    Xa = np.vstack([X, noise])
    ya = np.concatenate([y, np.full(n_noise, -1, dtype=int)])
    perm = rng.permutation(len(Xa))
    return Xa[perm], ya[perm]


def _gaussian_additive_noise(X: np.ndarray, sigma: float,
                             seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return X + sigma * rng.normal(size=X.shape) * X.std(axis=0)


def _touching_gaussians(n: int, seed: int, offset: float = 4.0,
                        D: int = 2, centers: int = 3):
    rng = np.random.default_rng(seed)
    per = max(1, n // centers)
    Xs, ys = [], []
    for i in range(centers):
        c = np.zeros(D); c[0] = i * offset
        Xs.append(rng.normal(loc=c, scale=1.0, size=(per, D)))
        ys.append(np.full(per, i, dtype=int))
    return np.vstack(Xs), np.concatenate(ys)


def _uneven_blobs(n: int, seed: int):
    rng = np.random.default_rng(seed)
    sizes = [int(0.6 * n), int(0.3 * n),
             max(1, n - int(0.6 * n) - int(0.3 * n))]
    centers = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
    stds = [0.5, 1.5, 0.8]
    Xs, ys = [], []
    for k, (s, c, sd) in enumerate(zip(sizes, centers, stds)):
        Xs.append(rng.normal(loc=c, scale=sd, size=(s, 2)))
        ys.append(np.full(s, k, dtype=int))
    return np.vstack(Xs), np.concatenate(ys)


def _helix_plane_sphere(n: int, seed: int):
    rng = np.random.default_rng(seed)
    per = n // 3
    t = np.linspace(0, 4 * np.pi, per)
    helix = np.stack([np.cos(t), np.sin(t), 0.3 * t], axis=1)
    helix += 0.05 * rng.normal(size=(per, 3))
    plane_xy = rng.uniform(-1.5, 1.5, size=(per, 2))
    plane = np.column_stack([plane_xy, np.full(per, 4.0)])
    plane += 0.05 * rng.normal(size=plane.shape)
    v = rng.normal(size=(per, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    sphere = 1.2 * v + np.array([5.0, 5.0, 0.0])
    sphere += 0.05 * rng.normal(size=(per, 3))
    X = np.vstack([helix, plane, sphere])
    y = np.concatenate([np.zeros(per), np.ones(per), 2 * np.ones(per)]).astype(int)
    return X, y


def _swiss_roll_2d(n: int, seed: int):
    """Two intertwined swiss-roll-like spirals embedded in 2D.

    The classic 'two_spirals' benchmark from manifold-clustering literature.
    """
    rng = np.random.default_rng(seed)
    per = n // 2
    t = np.linspace(0.5, 4.5, per) * np.pi
    spiral1 = np.column_stack([t * np.cos(t), t * np.sin(t)])
    spiral2 = np.column_stack([-t * np.cos(t), -t * np.sin(t)])
    spiral1 += 0.5 * rng.normal(size=spiral1.shape)
    spiral2 += 0.5 * rng.normal(size=spiral2.shape)
    X = np.vstack([spiral1, spiral2])
    y = np.concatenate([np.zeros(per), np.ones(per)]).astype(int)
    return X, y


def _swiss_roll_lifted(n: int, seed: int, D_target: int = 20):
    """3D swiss roll, partitioned into 4 angular bands, then linearly lifted
    to D_target dimensions via a random orthogonal embedding."""
    X3, t = make_swiss_roll(n_samples=n, noise=0.2, random_state=seed)
    bands = np.linspace(t.min(), t.max(), 5)
    y = np.digitize(t, bands[1:-1])
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(3, D_target))
    Q, _ = np.linalg.qr(A.T)
    return X3 @ Q[:3, :], y


def _imbalanced_blobs(n: int, seed: int, ratios, centers=None,
                      stds=None, D: int = 2, separation: float = 6.0):
    """Gaussian blobs with explicit per-cluster sample fractions.

    `ratios` is e.g. [0.9, 0.1] or [0.8, 0.15, 0.05]. Centers default to
    a regular configuration on a circle in the first two dims; remaining
    dims are zero. Std defaults to 1.0 for each cluster.
    """
    rng = np.random.default_rng(seed)
    K = len(ratios)
    ratios = np.asarray(ratios, dtype=float) / sum(ratios)
    sizes = (ratios * n).astype(int)
    sizes[-1] = n - sizes[:-1].sum()  # absorb rounding into last cluster
    if centers is None:
        theta = np.linspace(0, 2 * np.pi, K, endpoint=False)
        centers = np.zeros((K, D))
        centers[:, 0] = separation * np.cos(theta)
        centers[:, 1] = separation * np.sin(theta)
    if stds is None:
        stds = [1.0] * K
    Xs, ys = [], []
    for k, (s, c, sd) in enumerate(zip(sizes, centers, stds)):
        if s <= 0:
            continue
        Xs.append(rng.normal(loc=c, scale=sd, size=(int(s), D)))
        ys.append(np.full(int(s), k, dtype=int))
    return np.vstack(Xs), np.concatenate(ys)


def _imbalanced_moons(n: int, seed: int, ratio: float = 0.8,
                      noise: float = 0.05):
    """Two moons with `ratio`/(1-ratio) sample split.

    sklearn's make_moons forces a near-50/50 split; this generator builds
    each moon independently so the imbalance is honest.
    """
    rng = np.random.default_rng(seed)
    n0 = int(round(ratio * n))
    n1 = n - n0
    # Upper moon
    t0 = rng.uniform(0, np.pi, size=n0)
    moon0 = np.column_stack([np.cos(t0), np.sin(t0)])
    # Lower moon
    t1 = rng.uniform(0, np.pi, size=n1)
    moon1 = np.column_stack([1 - np.cos(t1), 0.5 - np.sin(t1)])
    X = np.vstack([moon0, moon1]) + noise * rng.normal(size=(n, 2))
    y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
    return X, y


def _hierarchical_clusters(n: int, seed: int, super_centers: int = 3,
                           sub_centers: int = 3, D: int = 2):
    """Each super-cluster is itself a Gaussian mixture of `sub_centers`
    sub-clusters. Useful for testing whether the algorithm finds the K=3 or
    K=9 partition (or both, via the bracket)."""
    rng = np.random.default_rng(seed)
    super_locs = rng.normal(scale=8.0, size=(super_centers, D))
    per_super = n // super_centers
    Xs, ys = [], []
    label = 0
    for s_loc in super_locs:
        sub_locs = s_loc + rng.normal(scale=1.5, size=(sub_centers, D))
        per_sub = per_super // sub_centers
        for sl in sub_locs:
            Xs.append(rng.normal(loc=sl, scale=0.3, size=(per_sub, D)))
            ys.append(np.full(per_sub, label, dtype=int))
            label += 1
    return np.vstack(Xs), np.concatenate(ys)


# ---------------------------------------------------------------------------
# Spec builders
# ---------------------------------------------------------------------------

def _spec(name: str, family: str, K_true: Optional[int],
          gen, n: int, *, noise_level: float = 0.0, noise_type: str = "",
          standardize: bool = True) -> DatasetSpec:
    return DatasetSpec(
        name=name, family=family, K_true=K_true,
        noise_level=noise_level, noise_type=noise_type,
        standardize_for_mbc=standardize,
        loader=lambda seed, _gen=gen, _n=n: _gen(_n, seed),
    )


def synth_specs(n: int) -> list:
    out = []

    # --- classic ----------------------------------------------------------
    out += [
        _spec("blobs_2D_std0.9", "classic", 4,
              lambda n, s: make_blobs(n_samples=n, centers=4, n_features=2,
                                      cluster_std=0.9, random_state=s), n),
        _spec("moons_n0.02", "classic", 2,
              lambda n, s: make_moons(n_samples=n, noise=0.02, random_state=s),
              n, noise_level=0.02, noise_type="gaussian"),
        _spec("circles_f0.5_n0.04", "classic", 2,
              lambda n, s: make_circles(n_samples=n, noise=0.04, factor=0.5,
                                        random_state=s),
              n, noise_level=0.04, noise_type="gaussian"),
    ]

    # --- noise_sweep ------------------------------------------------------
    for nz in (0.05, 0.10, 0.15):
        out.append(_spec(f"moons_n{nz:.2f}", "noise_sweep", 2,
                         lambda n, s, nz=nz: make_moons(
                             n_samples=n, noise=nz, random_state=s),
                         n, noise_level=nz, noise_type="gaussian"))
    for nz in (0.08, 0.15):
        out.append(_spec(f"circles_n{nz:.2f}", "noise_sweep", 2,
                         lambda n, s, nz=nz: make_circles(
                             n_samples=n, noise=nz, factor=0.5,
                             random_state=s),
                         n, noise_level=nz, noise_type="gaussian"))

    # --- bg_noise (uniform background contamination) ---------------------
    for frac in (0.05, 0.10, 0.20):
        def gen_blobs_bg(n, s, frac=frac):
            X, y = make_blobs(n_samples=int(n * (1 - frac)), centers=4,
                              n_features=2, cluster_std=0.9, random_state=s)
            return _add_uniform_bg_noise(X, y, frac, s)
        out.append(_spec(f"blobs2D_bg{int(frac*100)}", "bg_noise", 4,
                         gen_blobs_bg, n, noise_level=frac,
                         noise_type="uniform_bg"))

        def gen_moons_bg(n, s, frac=frac):
            X, y = make_moons(n_samples=int(n * (1 - frac)),
                              noise=0.05, random_state=s)
            return _add_uniform_bg_noise(X, y, frac, s)
        out.append(_spec(f"moons_bg{int(frac*100)}", "bg_noise", 2,
                         gen_moons_bg, n, noise_level=frac,
                         noise_type="uniform_bg"))

    # --- varied / anisotropic --------------------------------------------
    out += [
        _spec("blobs_2D_aniso", "varied", 4,
              lambda n, s: _aniso_blobs(n, s, centers=4, std=1.0, D=2), n),
        _spec("blobs_2D_varstd", "varied", 3,
              lambda n, s: make_blobs(n_samples=n, centers=3, n_features=2,
                                      cluster_std=[1.0, 2.5, 0.5],
                                      random_state=s), n),
        _spec("two_spirals_2D", "varied", 2,
              lambda n, s: _swiss_roll_2d(n, s), n,
              noise_level=0.5, noise_type="gaussian"),
    ]

    # --- high_D ----------------------------------------------------------
    for D in (50, 100, 200):
        out.append(_spec(f"blobs_{D}D_easy", "high_D", 6,
                         lambda n, s, D=D: make_blobs(
                             n_samples=n, centers=6, n_features=D,
                             cluster_std=3.0, random_state=s), n))
        out.append(_spec(f"blobs_{D}D_hard", "high_D", 6,
                         lambda n, s, D=D: make_blobs(
                             n_samples=n, centers=6, n_features=D,
                             cluster_std=6.0, random_state=s), n))
    out.append(_spec("blobs_50D_aniso", "high_D", 6,
                     lambda n, s: _aniso_blobs(n, s, centers=6, std=2.0, D=50),
                     n))
    out.append(_spec("swiss_roll_in_20D", "high_D", 4,
                     lambda n, s: _swiss_roll_lifted(n, s, D_target=20), n))

    # --- hierarchical ----------------------------------------------------
    out.append(_spec("hier_3x3_2D", "hierarchical", 9,
                     lambda n, s: _hierarchical_clusters(
                         n, s, super_centers=3, sub_centers=3, D=2), n))
    out.append(_spec("hier_3x3_10D", "hierarchical", 9,
                     lambda n, s: _hierarchical_clusters(
                         n, s, super_centers=3, sub_centers=3, D=10), n))

    # --- adversarial -----------------------------------------------------
    out += [
        _spec("touching_gauss_2D", "adversarial", 3,
              lambda n, s: _touching_gaussians(n, s, offset=4.0, D=2), n),
        _spec("touching_gauss_10D", "adversarial", 3,
              lambda n, s: _touching_gaussians(n, s, offset=4.0, D=10), n),
        _spec("uneven_blobs_2D", "adversarial", 3,
              lambda n, s: _uneven_blobs(n, s), n),
        _spec("helix_plane_sphere", "adversarial", 3,
              lambda n, s: _helix_plane_sphere(n, s), n),
    ]

    # --- imbalanced (biased cluster-size mixtures) ----------------------
    # Real datasets are rarely 50/50. These probe whether the bracket and
    # K-selection rule survive moderate-to-extreme class imbalance.
    out += [
        _spec("blobs_2D_imbal_90_10", "imbalanced", 2,
              lambda n, s: _imbalanced_blobs(n, s, [0.9, 0.1]), n),
        _spec("blobs_2D_imbal_80_15_5", "imbalanced", 3,
              lambda n, s: _imbalanced_blobs(n, s, [0.8, 0.15, 0.05]), n),
        _spec("blobs_2D_imbal_60_30_10", "imbalanced", 3,
              lambda n, s: _imbalanced_blobs(n, s, [0.6, 0.3, 0.1]), n),
        _spec("blobs_2D_imbal_4cluster", "imbalanced", 4,
              lambda n, s: _imbalanced_blobs(n, s, [0.5, 0.3, 0.15, 0.05]),
              n),
        _spec("blobs_50D_imbal_85_10_5", "imbalanced", 3,
              lambda n, s: _imbalanced_blobs(
                  n, s, [0.85, 0.10, 0.05], D=50, separation=8.0), n),
        _spec("moons_imbal_80_20", "imbalanced", 2,
              lambda n, s: _imbalanced_moons(n, s, ratio=0.8, noise=0.05),
              n),
        # Combined imbalance + bg noise — closest to the messy real-world
        # case (one majority cluster, a couple of minorities, plus dust).
        _spec("blobs_2D_imbal_with_bg10", "imbalanced", 3,
              lambda n, s: _add_uniform_bg_noise(
                  *_imbalanced_blobs(int(n*0.9), s, [0.7, 0.2, 0.1]),
                  frac=0.10/0.9, seed=s), n,
              noise_level=0.10, noise_type="uniform_bg"),
    ]

    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--n", type=int, default=2000,
                    help="Samples per synthetic dataset.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir or "results/synth")
    writer = RunWriter(out_dir=out_dir, suite="synth")

    specs = filter_specs(synth_specs(args.n), args.datasets, args.families)
    if not specs:
        print("No datasets selected after filtering."); return

    dtm_choices = dtm_modes(args.dtm)
    print(f"Synth suite: {len(specs)} datasets x {len(args.seeds)} seeds, "
          f"DTM={args.dtm}, n={args.n}, bracket_eval={args.bracket_eval}")

    rows = []
    for seed in args.seeds:
        for spec in specs:
            data = spec.loader(seed)
            if data is None:
                print(f"\n=== {spec.name} seed={seed}: not available ==="); continue
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
