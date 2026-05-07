#!/usr/bin/env python3
"""
mbc_runner.py — shared library for run_synth.py, run_real.py, run_neuro.py.

Provides:
  - param presets (with/without DTM rescue)
  - MBC + bracket-K evaluation, baselines (DBSCAN/HDBSCAN with grids,
    OPTICS, BIRCH, KMeans, GMM, Spectral, Ward) — all returning row dicts
  - score_pack: ARI, NMI, silhouette, K
  - preprocessing helpers (z-score, PCA, 2D projection)
  - multi-panel scatter plotter
  - RunWriter: CSV + markdown report per suite, faceted by metadata

Row metadata columns (carried through to CSVs):
  dataset, family, D, n, K_true, noise_level, noise_type, seed, algo, K,
  bracket_low, bracket_high, ARI, NMI, Sil, runtime,
  k_star, k_low, k_high, regime, dtm_rescue_used, n_rescued_edges,
  d_eff, noise_count, n_edges_final, K_eval_kind
  (K_eval_kind in {"main","bracket_canonical","bracket_mid","bracket_max"})
"""
from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import (
    DBSCAN,
    OPTICS,
    Birch,
    KMeans,
    SpectralClustering,
    AgglomerativeClustering,
)
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from MBC import MBCParams, mbc_cluster, standardize_then_mbc, get_labels_for_K

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except Exception:
    HDBSCAN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Param presets — only DTM is toggled at the runner level.
# ---------------------------------------------------------------------------

def make_params(use_dtm: bool, **overrides: Any) -> MBCParams:
    p = MBCParams()
    p.use_dtm_rescue = bool(use_dtm)
    for k, v in overrides.items():
        if not hasattr(p, k):
            raise AttributeError(f"MBCParams has no field {k!r}")
        setattr(p, k, v)
    return p


def dtm_modes(spec: str) -> List[bool]:
    spec = spec.lower()
    if spec == "on":   return [True]
    if spec == "off":  return [False]
    if spec == "both": return [True, False]
    raise ValueError(f"--dtm must be on|off|both, got {spec!r}")


def mbc_label(use_dtm: bool, K_eval_kind: str = "main") -> str:
    base = "MBC+DTM" if use_dtm else "MBC"
    if K_eval_kind == "main":
        return base
    suffix = {
        "bracket_low":  "[K_low]",
        "bracket_mid":  "[K_mid]",
        "bracket_high": "[K_high]",
    }.get(K_eval_kind, f"[{K_eval_kind}]")
    return f"{base}{suffix}"


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def zscore(X: np.ndarray) -> np.ndarray:
    return StandardScaler(with_std=True).fit_transform(np.asarray(X, float))


def pca_if_needed(X: np.ndarray, target_dim: int, seed: int = 0) -> np.ndarray:
    X = np.asarray(X, float)
    if X.shape[1] <= target_dim:
        return zscore(X)
    Xs = zscore(X)
    return PCA(n_components=target_dim, random_state=seed).fit_transform(Xs)


def project_2d(X: np.ndarray, seed: int = 0) -> np.ndarray:
    X = np.asarray(X, float)
    if X.shape[1] == 2:
        return X
    return PCA(n_components=2, random_state=seed).fit_transform(X)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def safe_silhouette(X: np.ndarray, labels: np.ndarray, max_n: int = 10_000) -> float:
    labs = np.asarray(labels)
    if len(labs) > max_n:
        return float("nan")
    mask = labs != -1
    if mask.sum() < 2:
        return float("nan")
    if len(np.unique(labs[mask])) < 2:
        return float("nan")
    try:
        return float(silhouette_score(X[mask], labs[mask]))
    except Exception:
        return float("nan")


def score_pack(
    X: np.ndarray, y_true: Optional[np.ndarray], labels: np.ndarray
) -> Dict[str, float]:
    labs = np.asarray(labels)
    n_clusters = int((np.unique(labs) != -1).sum())
    ari = adjusted_rand_score(y_true, labs) if y_true is not None else float("nan")
    nmi = normalized_mutual_info_score(y_true, labs) if y_true is not None else float("nan")
    return dict(K=n_clusters, ARI=float(ari), NMI=float(nmi),
                Sil=safe_silhouette(X, labs))


# ---------------------------------------------------------------------------
# MBC + bracket-K evaluation
# ---------------------------------------------------------------------------

@dataclass
class MBCRow:
    """One MBC evaluation point (canonical or bracket K)."""
    algo: str
    labels: np.ndarray
    K_eval_kind: str
    K_target: Optional[int]
    runtime: float
    info: Dict[str, Any]


def _pick_low_mid_high(Ks_sorted: List[int],
                       formal_low: int,
                       formal_high: int,
                       canonical: int) -> List[Tuple[str, int]]:
    """Choose bracket K's for the low/mid/high panel triplet.

    Restrict to Ks within the formal bracket [formal_low, formal_high]
    when possible (so we don't pick stray sweep K's far outside the bracket).
    Dedupe against the canonical K. Return list of (kind, K) up to 3 entries.
    """
    in_band = [K for K in Ks_sorted if formal_low <= K <= formal_high]
    pool = in_band if len(in_band) >= 1 else list(Ks_sorted)
    if not pool:
        return []
    K_low = pool[0]
    K_high = pool[-1]
    K_mid = pool[len(pool) // 2]
    candidates = [("bracket_low", K_low), ("bracket_mid", K_mid),
                  ("bracket_high", K_high)]
    # Drop anything matching canonical (already covered by main row); dedupe.
    seen: set = {canonical}
    out: List[Tuple[str, int]] = []
    for kind, K in candidates:
        if K in seen:
            continue
        out.append((kind, int(K)))
        seen.add(K)
    return out


def run_mbc(
    X: np.ndarray,
    use_dtm: bool,
    seed: int = 0,
    standardize: bool = True,
    bracket_eval: str = "canonical",
    **param_overrides: Any,
) -> List[MBCRow]:
    """Run MBC and return one or more rows.

    bracket_eval:
      "canonical"    -> 1 row: the main labels (K = res.n_clusters).
      "low_mid_high" -> up to 4 rows: canonical + (low, mid, high) chosen from
                        bracket_labels keys filtered to the formal bracket.
      "all"          -> 1 + N rows, one per distinct K in res.bracket_labels.
    """
    p = make_params(use_dtm=use_dtm, **param_overrides)
    fn = standardize_then_mbc if standardize else mbc_cluster
    t0 = time.time()
    res = fn(np.asarray(X, float), p, seed=seed)
    t = time.time() - t0

    bm = res.info.get("bracket_mass", (res.bracket[0], res.bracket[1]))
    bmp = res.info.get("bracket_mass_persistent", bm)
    base_info = dict(
        bracket_low=int(res.bracket[0]),
        bracket_high=int(res.bracket[1]),
        bracket_mass_low=int(bm[0]),
        bracket_mass_high=int(bm[1]),
        bracket_mass_persistent_low=int(bmp[0]),
        bracket_mass_persistent_high=int(bmp[1]),
        mass_gamma=float(res.info.get("mass_gamma", float("nan"))),
        k_star=int(res.k_star),
        k_low=int(res.k_low),
        k_high=int(res.k_high),
        rho_hat=float(res.rho_hat),
        regime=str(res.info.get("regime", "")),
        dtm_rescue_used=bool(res.dtm_rescue_used),
        n_rescued_edges=int(res.n_rescued_edges),
        d_eff=int(res.d_eff),
        n_edges_final=int(res.n_edges_final),
        noise_count=int(res.noise_count),
        bracket_K_curve=str(res.bracket_K_curve),
        bracket_Ks_avail=str(sorted(res.bracket_labels.keys())),
    )

    rows: List[MBCRow] = [MBCRow(
        algo=mbc_label(use_dtm, "main"),
        labels=res.labels, K_eval_kind="main",
        K_target=int(res.n_clusters), runtime=t, info=dict(base_info),
    )]

    if bracket_eval == "canonical":
        return rows

    Ks = sorted(res.bracket_labels.keys())
    if not Ks:
        return rows

    if bracket_eval == "all":
        for K in Ks:
            rows.append(MBCRow(
                algo=f"{mbc_label(use_dtm, 'main')}[K={K}]",
                labels=res.bracket_labels[K],
                K_eval_kind=f"bracket_K{K}",
                K_target=int(K), runtime=t, info=dict(base_info),
            ))
        return rows

    # low_mid_high
    chosen = _pick_low_mid_high(
        Ks, res.bracket[0], res.bracket[1], int(res.n_clusters))
    for kind, K in chosen:
        labs = res.bracket_labels[K]
        rows.append(MBCRow(
            algo=mbc_label(use_dtm, kind),
            labels=labs, K_eval_kind=kind,
            K_target=int(K), runtime=t, info=dict(base_info),
        ))
    return rows


# ---------------------------------------------------------------------------
# Baselines — return list of (algo_name, labels, runtime).
# ---------------------------------------------------------------------------

def _kdist_eps(X: np.ndarray, k: int) -> float:
    """k-distance heuristic for DBSCAN eps."""
    n = len(X)
    algo = "brute" if X.shape[1] >= 30 else "auto"
    nn = NearestNeighbors(n_neighbors=min(k + 1, n), algorithm=algo,
                          metric="euclidean").fit(X)
    dists, _ = nn.kneighbors(X, return_distance=True)
    return float(np.median(np.sort(dists[:, k])))


@dataclass
class BaselineRow:
    """One baseline run; `in_panel` controls whether it appears in the grid."""
    algo: str
    labels: np.ndarray
    runtime: float
    in_panel: bool = False


def run_baselines(
    X: np.ndarray,
    seed: int = 0,
    K_true: Optional[int] = None,
) -> List[BaselineRow]:
    """Run baseline clustering algorithms.

    Returns a flat list of BaselineRow. Every grid point is in the list (so
    CSV captures the full sweep), but only one representative per algo family
    has `in_panel=True` so the panel grid stays clean.

    Panel selection per family:
      DBSCAN   -> default sklearn settings
      HDBSCAN  -> default cluster_selection_method='eom', min_cluster_size=2% of n
      OPTICS   -> default
      BIRCH    -> default
      KMeans   -> at K_true (if known), else K=3
      GMM      -> at K_true (if known), else K=3
      Ward     -> at K_true (if known), else K=3
      Spectral -> at K_true (if known), else K=3 — skipped for n>3000
    """
    n = len(X)
    out: List[BaselineRow] = []

    def _run(name: str, fn: Callable[[], np.ndarray],
             in_panel: bool = False) -> None:
        try:
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                labels = fn()
            out.append(BaselineRow(
                algo=name, labels=np.asarray(labels),
                runtime=time.time() - t0, in_panel=in_panel))
        except Exception as e:
            print(f"    [{name}] skipped: {e}")

    # ----- DBSCAN: default (panel) + grid (CSV only) -----
    _run("DBSCAN(default)", lambda: DBSCAN().fit_predict(X), in_panel=True)
    try:
        k_for_eps = max(5, int(5 * math.log(max(3, n))))
        eps0 = _kdist_eps(X, k_for_eps)
        for f in (0.7, 1.0, 1.5):
            for ms in (5, k_for_eps):
                tag = f"DBSCAN(eps={f:.1f}*kd,ms={ms})"
                _run(tag, lambda f=f, ms=ms: DBSCAN(
                    eps=max(1e-9, f * eps0), min_samples=ms).fit_predict(X))
    except Exception as e:
        print(f"    [DBSCAN grid] skipped: {e}")

    # ----- OPTICS, BIRCH -----
    _run("OPTICS", lambda: OPTICS().fit_predict(X), in_panel=True)
    _run("BIRCH",  lambda: Birch().fit_predict(X),  in_panel=True)

    # ----- KMeans, GMM, Ward, Spectral: K_true (panel) + grid (CSV) -----
    K_panel = int(K_true) if (K_true is not None and 2 <= K_true < n) else 3
    K_grid: List[int] = [K_panel]
    for k in (3, 8):
        if k != K_panel and 2 <= k < n:
            K_grid.append(k)

    for K in K_grid:
        is_panel = (K == K_panel)
        # No "@K=" suffix for the panel version when K_true is known and == K.
        if K_true is not None and K == K_true:
            tag_k = ""
        elif is_panel:
            tag_k = f"@K={K}"
        else:
            tag_k = f"@K={K}"
        _run(f"KMeans{tag_k}", lambda K=K: KMeans(
            n_clusters=K, n_init=10, random_state=seed).fit_predict(X),
            in_panel=is_panel)
        _run(f"GMM{tag_k}", lambda K=K: GaussianMixture(
            n_components=K, covariance_type="full",
            random_state=seed, max_iter=200).fit(X).predict(X),
            in_panel=is_panel)
        _run(f"Ward{tag_k}", lambda K=K: AgglomerativeClustering(
            n_clusters=K, linkage="ward").fit_predict(X),
            in_panel=is_panel)
        if n <= 3000:
            _run(f"Spectral{tag_k}", lambda K=K: SpectralClustering(
                n_clusters=K, affinity="nearest_neighbors",
                assign_labels="kmeans", random_state=seed,
                n_neighbors=min(10, n - 1)).fit_predict(X),
                in_panel=is_panel)

    # ----- HDBSCAN: default eom (panel) + grid over min_cluster_size (CSV) -----
    if HDBSCAN_AVAILABLE:
        # Panel = HDBSCAN's "recommended" output: EOM selection at a sensible
        # min_cluster_size (2% of n, floor 5). This is what HDBSCAN authors
        # surface as the default partition.
        mcs_panel = max(5, int(0.02 * n))
        _run(f"HDBSCAN(eom,mcs=2%)",
             lambda mcs=mcs_panel: hdbscan.HDBSCAN(
                 min_cluster_size=mcs,
                 cluster_selection_method="eom").fit_predict(X),
             in_panel=True)
        # Grid: vary min_cluster_size and selection method (CSV only).
        for mcs_frac in (0.005, 0.01, 0.05):
            mcs = max(5, int(mcs_frac * n))
            _run(f"HDBSCAN(eom,mcs={mcs_frac:.0%})",
                 lambda mcs=mcs: hdbscan.HDBSCAN(
                     min_cluster_size=mcs,
                     cluster_selection_method="eom").fit_predict(X))
        for mcs_frac in (0.01, 0.02, 0.05):
            mcs = max(5, int(mcs_frac * n))
            _run(f"HDBSCAN(leaf,mcs={mcs_frac:.0%})",
                 lambda mcs=mcs: hdbscan.HDBSCAN(
                     min_cluster_size=mcs,
                     cluster_selection_method="leaf").fit_predict(X))

    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

@dataclass
class Panel:
    title: str
    labels: np.ndarray
    metrics: Dict[str, float] = field(default_factory=dict)
    section: Optional[str] = None  # "MBC" | "density" | "partitional"


def _palette_colors(K: int):
    """Return a list of K visually distinct colors. Uses tab20 up to 20, then
    falls back to a continuous colormap (turbo) for high-K labelings."""
    if K <= 20:
        cmap = plt.get_cmap("tab20")
        return [cmap(i) for i in range(K)]
    cmap = plt.get_cmap("turbo")
    return [cmap(i / max(1, K - 1)) for i in range(K)]


def _scatter(ax, X2: np.ndarray, labels: np.ndarray, title: str, sub: str = "",
             border_color: Optional[str] = None):
    labs = np.asarray(labels)
    u = np.unique(labs)
    pos = [v for v in u if v != -1]
    colors = _palette_colors(max(1, len(pos)))
    color_for = {int(v): colors[i] for i, v in enumerate(pos)}

    for v in u:
        m = labs == v
        if v == -1:
            ax.scatter(X2[m, 0], X2[m, 1], s=3, alpha=0.35, c="0.7",
                       marker="x", linewidths=0.6)
        else:
            ax.scatter(X2[m, 0], X2[m, 1], s=6, alpha=0.85,
                       c=[color_for[int(v)]], linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    if sub:
        ax.text(0.02, 0.02, sub, transform=ax.transAxes, fontsize=7.5,
                alpha=0.85,
                bbox=dict(boxstyle="round,pad=0.18", fc="white",
                          ec="none", alpha=0.65))
    if border_color is not None:
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.0)


SECTION_COLORS = {
    "MBC":         "#1f77b4",  # blue
    "density":     "#2ca02c",  # green
    "partitional": "#9467bd",  # purple
}


def _section_order(panels: Sequence[Panel]) -> List[str]:
    """Return distinct sections in the order they first appear."""
    seen: List[str] = []
    for p in panels:
        s = getattr(p, "section", None) or ""
        if s and s not in seen:
            seen.append(s)
    return seen


def plot_panel(
    X: np.ndarray,
    panels: Sequence[Panel],
    out_path: Path,
    suptitle: str = "",
    seed: int = 0,
    max_cols: int = 4,
) -> None:
    """Render a faceted grid of clustering panels, breaking to a new row at
    each section boundary so MBC / density / partitional are visually
    grouped. Each panel's section determines its border color.
    """
    if not panels:
        return
    X2 = project_2d(X, seed=seed)

    # Group panels by section (first-seen order); panels with no section
    # land in a final "_unclassified" row.
    sections = _section_order(panels) or ["_unclassified"]
    grouped: Dict[str, List[Panel]] = {s: [] for s in sections}
    for p in panels:
        s = getattr(p, "section", None) or "_unclassified"
        grouped.setdefault(s, []).append(p)

    cols = max(1, min(max_cols,
                      max(len(v) for v in grouped.values() if v)))
    rows = sum((len(v) + cols - 1) // cols for v in grouped.values() if v)

    fig_w = 3.2 * cols + 0.4
    fig_h = 3.0 * rows + (0.7 if suptitle else 0.2)
    fig, axes = plt.subplots(
        rows, cols, figsize=(fig_w, fig_h), dpi=150,
        constrained_layout=False,
        gridspec_kw=dict(top=(1 - 0.45 / fig_h) if suptitle else 0.97,
                         bottom=0.04, left=0.02, right=0.99,
                         hspace=0.35, wspace=0.10),
    )
    axes = np.atleast_2d(axes).reshape(rows, cols)

    flat_idx = 0
    for s, group in grouped.items():
        if not group:
            continue
        section_rows = (len(group) + cols - 1) // cols
        for j, pn in enumerate(group):
            r = flat_idx // cols + j // cols
            c = j % cols
            sub_bits = []
            for key, fmt in (("ARI", "{:.3f}"), ("NMI", "{:.3f}"),
                             ("Sil", "{:.2f}")):
                v = pn.metrics.get(key)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    sub_bits.append(f"{key}={fmt.format(v)}")
            border = SECTION_COLORS.get(s)
            _scatter(axes[r, c], X2, pn.labels, title=pn.title,
                     sub="  ".join(sub_bits), border_color=border)
        # Hide any unused cells in the final row of this section.
        for j in range(len(group), section_rows * cols):
            r = flat_idx // cols + j // cols
            c = j % cols
            axes[r, c].axis("off")
        flat_idx += section_rows * cols

    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=0.995)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    """A single benchmark dataset; the loader returns (X, y_or_None)."""
    name: str
    loader: Callable[[int], Optional[Tuple[np.ndarray, Optional[np.ndarray]]]]
    family: str = ""
    K_true: Optional[int] = None
    noise_level: float = 0.0
    noise_type: str = ""
    standardize_for_mbc: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


def _meta_row(spec: DatasetSpec, X: np.ndarray, seed: int) -> Dict[str, Any]:
    return dict(
        dataset=spec.name, family=spec.family,
        D=int(X.shape[1]), n=int(X.shape[0]),
        K_true=spec.K_true if spec.K_true is not None else "",
        noise_level=spec.noise_level, noise_type=spec.noise_type,
        seed=seed,
    )


_DENSITY_FAMILY = ("DBSCAN", "HDBSCAN", "OPTICS", "BIRCH")


def _baseline_section(algo: str) -> str:
    return "density" if algo.startswith(_DENSITY_FAMILY) else "partitional"


def run_dataset(
    spec: DatasetSpec,
    X: np.ndarray,
    y: Optional[np.ndarray],
    seed: int,
    dtm_choices: Sequence[bool],
    figs_dir: Path,
    bracket_eval: str = "canonical",
    skip_baselines: bool = False,
) -> List[Dict[str, Any]]:
    """Run MBC (canonical + optional bracket-K extras) and baselines on (X, y).

    Records every row to the CSV, but the saved panel figure is curated:
      MBC group       (canonical + bracket low/mid/high if requested)
      density group   (DBSCAN, HDBSCAN, OPTICS, BIRCH — defaults only)
      partitional grp (KMeans, GMM, Ward, Spectral — at K_true if known)
    """
    figs_dir.mkdir(parents=True, exist_ok=True)
    Xs_for_metrics = zscore(np.asarray(X, float))

    mbc_panels: List[Panel] = []
    density_panels: List[Panel] = []
    partitional_panels: List[Panel] = []
    rows: List[Dict[str, Any]] = []
    meta = _meta_row(spec, X, seed)

    # ---- MBC (every requested DTM mode and bracket K) ----
    for use_dtm in dtm_choices:
        try:
            mbc_rows = run_mbc(
                X, use_dtm=use_dtm, seed=seed,
                standardize=spec.standardize_for_mbc,
                bracket_eval=bracket_eval,
            )
        except Exception as e:
            print(f"  [MBC dtm={use_dtm}] failed: {e}")
            continue
        for mr in mbc_rows:
            sc = score_pack(Xs_for_metrics, y, mr.labels)
            row = dict(meta)
            row.update(dict(
                algo=mr.algo, K_eval_kind=mr.K_eval_kind,
                K=sc["K"], ARI=sc["ARI"], NMI=sc["NMI"], Sil=sc["Sil"],
                runtime=mr.runtime,
            ))
            row.update(mr.info)
            rows.append(row)

            # Panel title: MBC variant + observed K and formal bracket
            title = (
                f"{mr.algo} K={sc['K']} "
                f"bracket=[{mr.info['bracket_low']},{mr.info['bracket_high']}]"
            )
            mbc_panels.append(Panel(title=title, labels=mr.labels,
                                    metrics=sc, section="MBC"))
            print(f"  {mr.algo:<18} K={sc['K']:<3} "
                  f"bracket=[{mr.info['bracket_low']},{mr.info['bracket_high']}] "
                  f"ARI={sc['ARI']:.3f} NMI={sc['NMI']:.3f} "
                  f"Sil={sc['Sil']:.3f} ({mr.runtime:.2f}s) "
                  f"regime={mr.info['regime']}")

    # ---- Baselines ----
    if not skip_baselines:
        for br in run_baselines(Xs_for_metrics, seed=seed, K_true=spec.K_true):
            sc = score_pack(Xs_for_metrics, y, br.labels)
            row = dict(meta)
            row.update(dict(algo=br.algo, K_eval_kind="baseline",
                            K=sc["K"], ARI=sc["ARI"], NMI=sc["NMI"],
                            Sil=sc["Sil"], runtime=br.runtime))
            rows.append(row)

            print(f"  {br.algo:<22} K={sc['K']:<3} "
                  f"ARI={sc['ARI']:.3f} NMI={sc['NMI']:.3f} "
                  f"Sil={sc['Sil']:.3f} ({br.runtime:.2f}s)"
                  f"{' [panel]' if br.in_panel else ''}")

            if br.in_panel:
                section = _baseline_section(br.algo)
                pn = Panel(title=f"{br.algo} K={sc['K']}",
                           labels=br.labels, metrics=sc, section=section)
                if section == "density":
                    density_panels.append(pn)
                else:
                    partitional_panels.append(pn)

    # ---- Panel grid (MBC row -> density row -> partitional row) ----
    panels = mbc_panels + density_panels + partitional_panels
    if panels:
        plot_panel(
            Xs_for_metrics, panels,
            out_path=figs_dir / f"{spec.name}_seed{seed}.png",
            suptitle=f"{spec.name} (n={len(X)}, d={X.shape[1]}"
                     + (f", K_true={spec.K_true}" if spec.K_true else "") + ")",
            seed=seed,
            max_cols=4,
        )

    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

REPORT_COLS = ["K", "ARI", "NMI", "Sil", "runtime"]


@dataclass
class RunWriter:
    out_dir: Path
    suite: str

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.figs_dir = self.out_dir / "figs"
        self.figs_dir.mkdir(parents=True, exist_ok=True)

    def _table_md(self, df: pd.DataFrame) -> str:
        try:
            return df.round(3).to_markdown()
        except ImportError:
            return "```\n" + df.round(3).to_string() + "\n```"

    def write(self, rows: List[Dict[str, Any]]) -> None:
        df = pd.DataFrame(rows)
        raw = self.out_dir / f"{self.suite}_raw.csv"
        summ = self.out_dir / f"{self.suite}_summary.csv"
        md = self.out_dir / f"{self.suite}_report.md"
        df.to_csv(raw, index=False)

        if df.empty:
            md.write_text(f"# {self.suite}\n\n(no rows)\n")
            return

        agg_cols = [c for c in REPORT_COLS if c in df.columns]
        keys = ["dataset", "algo"]
        g = df.groupby(keys)[agg_cols].mean(numeric_only=True)
        g.to_csv(summ)

        with md.open("w") as f:
            f.write(f"# {self.suite}\n\n")
            f.write(f"- raw: `{raw.name}` ({len(df)} rows)\n")
            f.write(f"- summary: `{summ.name}`\n")
            f.write(f"- figs/: per-dataset multi-panel scatter\n\n")

            # Per-family breakdown for synth (or any suite with a 'family' col)
            if "family" in df.columns and df["family"].notna().any() \
                    and df["family"].astype(str).str.len().sum() > 0:
                fams = sorted(x for x in df["family"].unique()
                              if isinstance(x, str) and x)
                for fam in fams:
                    sub = df[df["family"] == fam]
                    if sub.empty:
                        continue
                    f.write(f"## family: {fam} ({len(sub)} rows)\n\n")
                    keys_fam = [c for c in ("dataset", "algo") if c in sub.columns]
                    g_fam = sub.groupby(keys_fam)[agg_cols].mean(
                        numeric_only=True)
                    f.write(self._table_md(g_fam) + "\n\n")

            f.write("## All rows by (dataset, algo)\n\n")
            f.write(self._table_md(g) + "\n")

        print(f"\nWrote {raw}")
        print(f"Wrote {summ}")
        print(f"Wrote {md}")


# ---------------------------------------------------------------------------
# CLI helper shared by all runners
# ---------------------------------------------------------------------------

def add_common_args(parser):
    parser.add_argument("--dtm", choices=["on", "off", "both"], default="off",
                        help="Run MBC with DTM rescue on, off, or both. "
                             "Default off — DTM rarely changes outputs.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7],
                        help="Random seeds (one row per seed per algo).")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory; defaults to results/<suite>.")
    parser.add_argument("--datasets", type=str, nargs="*", default=None,
                        help="Substring filter on dataset name "
                             "(case-insensitive).")
    parser.add_argument("--families", type=str, nargs="*", default=None,
                        help="Substring filter on family.")
    parser.add_argument("--no_baselines", action="store_true",
                        help="Skip baseline algorithms.")
    parser.add_argument("--bracket_eval",
                        choices=["canonical", "low_mid_high", "all"],
                        default="low_mid_high",
                        help="Bracket-K evaluation: "
                             "canonical=just main K, "
                             "low_mid_high=add low/mid/high K from formal bracket "
                             "(default), "
                             "all=add every distinct bracket K.")


def filter_specs(specs: Sequence[DatasetSpec],
                 name_patterns: Optional[Iterable[str]],
                 family_patterns: Optional[Iterable[str]] = None
                 ) -> List[DatasetSpec]:
    out = list(specs)
    if name_patterns:
        pats = [p.lower() for p in name_patterns]
        out = [s for s in out if any(p in s.name.lower() for p in pats)]
    if family_patterns:
        fpats = [p.lower() for p in family_patterns]
        out = [s for s in out if any(p in s.family.lower() for p in fpats)]
    return out
