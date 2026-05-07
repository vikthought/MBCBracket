#!/usr/bin/env python3
"""
run_neuro.py — neuroscience benchmark for MBC.

Datasets (loaded from data/):
  Retina_full           all RGCs (some unlabeled, label 0)
  Retina_labeled        labeled cells only (label > 0)
  Retina_off_brisk      off_brisk_sustained + off_brisk_transient
  Retina_non_off_brisk  labeled cells minus off-brisk subset
  V1                    V1 diffusion-map embedding (no GT)

The diffusion-map embeddings are already preprocessed; the runner skips the
default z-score+PCA wrapper for these (sets standardize_for_mbc=False).

Because MBC tends to report K=1 for diffmap embeddings, this runner defaults
to `--bracket_eval mid_max` so the report includes ARI/NMI/Sil for the
canonical K and for two additional K's chosen from the bracket (middle and
maximum), with the partition for each plotted side-by-side.

Usage:
    python run_neuro.py
    python run_neuro.py --datasets Retina_labeled --bracket_eval all
    python run_neuro.py --no_baselines --seeds 7 11 23
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from mbc_runner import (
    DatasetSpec,
    RunWriter,
    add_common_args,
    dtm_modes,
    filter_specs,
    run_dataset,
)

DATA_DIR = Path("data")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _retina_raw():
    Psi = np.load(DATA_DIR / "retina_diffmap.npy")
    rgc = np.load(DATA_DIR / "rgc_types.npy", allow_pickle=True)
    types = np.array([str(t) for t in rgc])
    type_to_int = {t: i for i, t in enumerate(sorted(np.unique(types)))}
    color_labels = np.array([type_to_int[t] for t in types], dtype=int)
    return Psi, types, color_labels


def _load_retina_full(seed: int):
    try:
        Psi, _, color_labels = _retina_raw()
    except Exception as e:
        print(f"  [Retina_full] load failed: {e}"); return None
    return Psi, color_labels


def _load_retina_labeled(seed: int):
    try:
        Psi, _, color_labels = _retina_raw()
    except Exception as e:
        print(f"  [Retina_labeled] load failed: {e}"); return None
    mask = color_labels > 0
    return Psi[mask], color_labels[mask]


def _load_retina_off_brisk(seed: int):
    try:
        Psi, types, _ = _retina_raw()
    except Exception as e:
        print(f"  [Retina_off_brisk] load failed: {e}"); return None
    mask = (types == "off_brisk_sustained") | (types == "off_brisk_transient")
    if mask.sum() == 0:
        return None
    sub = types[mask]
    type_to_int = {t: i for i, t in enumerate(sorted(np.unique(sub)))}
    y = np.array([type_to_int[t] for t in sub], dtype=int)
    return Psi[mask], y


def _load_retina_non_off_brisk(seed: int):
    try:
        Psi, types, color_labels = _retina_raw()
    except Exception as e:
        print(f"  [Retina_non_off_brisk] load failed: {e}"); return None
    labeled = color_labels > 0
    is_off_brisk = (types == "off_brisk_sustained") | (types == "off_brisk_transient")
    mask = labeled & ~is_off_brisk
    if mask.sum() == 0:
        return None
    sub = types[mask]
    type_to_int = {t: i for i, t in enumerate(sorted(np.unique(sub)))}
    y = np.array([type_to_int[t] for t in sub], dtype=int)
    return Psi[mask], y


def _load_v1(seed: int):
    path = DATA_DIR / "V1-diffmap.npy"
    if not path.exists():
        return None
    try:
        Psi = np.load(path)
    except Exception as e:
        print(f"  [V1] load failed: {e}"); return None
    return Psi, None


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------

def neuro_specs() -> list:
    common = dict(family="neuro_diffmap", standardize_for_mbc=False)
    return [
        DatasetSpec(name="Retina_full", K_true=None,
                    loader=_load_retina_full, **common),
        DatasetSpec(name="Retina_labeled", K_true=None,
                    loader=_load_retina_labeled, **common),
        DatasetSpec(name="Retina_off_brisk", K_true=2,
                    loader=_load_retina_off_brisk, **common),
        DatasetSpec(name="Retina_non_off_brisk", K_true=None,
                    loader=_load_retina_non_off_brisk, **common),
        DatasetSpec(name="V1", K_true=None,
                    loader=_load_v1, **common),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()

    out_dir = Path(args.out_dir or "results/neuro")
    writer = RunWriter(out_dir=out_dir, suite="neuro")

    specs = filter_specs(neuro_specs(), args.datasets, args.families)
    if not specs:
        print("No datasets selected after filtering."); return

    dtm_choices = dtm_modes(args.dtm)
    print(f"Neuro suite: {len(specs)} datasets x {len(args.seeds)} seeds, "
          f"DTM={args.dtm}, bracket_eval={args.bracket_eval}")

    rows = []
    for seed in args.seeds:
        for spec in specs:
            data = spec.loader(seed)
            if data is None:
                print(f"\n=== {spec.name} seed={seed}: not available, skipping ===")
                continue
            X, y = data
            print(f"\n=== {spec.name} [{spec.family}] seed={seed} "
                  f"(n={len(X)}, D={X.shape[1]}"
                  f"{', no GT' if y is None else ''}) ===")
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
