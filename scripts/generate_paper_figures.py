#!/usr/bin/env python3
"""Generate the empirical figures for PAPER/main.tex.

Outputs go to PAPER/figures/empirical/ as both .pdf and .png.

Figures:
    fig_per_family_brackets.{pdf,png}      — 2x4 small multiples of bracket
                                             widths, one panel per synth family
    fig_per_regime_brackets.{pdf,png}      — 1x3 boxplot by MBC regime
                                             (separable / transitional / non-separable)
    fig_perturbation_sweep.{pdf,png}       — three controlled perturbations
                                             (bg contamination / scale ratio /
                                              cluster offset) on blobs
    fig_K_recovery_scatter.{pdf,png}       — K_practical vs K_true scatter
                                             with bracket whiskers, colored
                                             by family
    fig_retina_filtration.{pdf,png}        — retina + V1 K(k) filtration
                                             with bracket band

Run:
    python scripts/generate_paper_figures.py
"""
from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RES = ROOT / "results"
OUT = ROOT / "PAPER" / "figures" / "empirical"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Bitstream Vera Serif", "Times New Roman"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "lines.linewidth": 1.2,
    "patch.linewidth": 0.7,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COL_MBC = "#C0392B"
COL_HDB = "#2C3E50"
COL_DBS = "#7F8C8D"
COL_KSTAR = "#27AE60"
PALETTE = ["#C0392B", "#2980B9", "#27AE60", "#8E44AD", "#E67E22",
           "#16A085", "#D35400", "#34495E", "#7D3C98", "#1ABC9C"]
FAMILY_COLORS = {
    "classic":      "#C0392B",
    "noise_sweep":  "#E67E22",
    "bg_noise":     "#D35400",
    "varied":       "#27AE60",
    "high_D":       "#2980B9",
    "hierarchical": "#8E44AD",
    "adversarial":  "#34495E",
    "imbalanced":   "#16A085",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_mode(s):
    s = s.dropna()
    if len(s) == 0:
        return None
    return s.mode().iloc[0]


def _load_curve(dataset: str, suite: str):
    df = pd.read_csv(RES / suite / f"{suite}_raw.csv")
    rows = df[(df["dataset"] == dataset) & (df["algo"] == "MBC")]
    if len(rows) == 0:
        return None
    r = rows.iloc[0]
    try:
        curve = ast.literal_eval(r["bracket_K_curve"])
    except Exception:
        curve = None
    return {
        "dataset": dataset,
        "curve": curve,
        "k_low": int(r["k_low"]),
        "k_high": int(r["k_high"]),
        "k_star": int(r["k_star"]),
        "bracket_low": int(r["bracket_low"]),
        "bracket_high": int(r["bracket_high"]),
        "rho": float(r["rho_hat"]),
        "regime": r["regime"],
        "K_practical": int(r["K"]),
        "ARI": r["ARI"],
        "n": int(r["n"]),
        "D": int(r["D"]),
    }


def _save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  wrote {OUT / name}.{{pdf,png}}")


# ---------------------------------------------------------------------------
# Fig 3a: per-family small multiples (8 panels)
# ---------------------------------------------------------------------------

def fig_per_family_brackets():
    """Per-family informativeness = coverage / (median width + 1).
    Higher = more committal answer that's still right.
    """
    print("Figure: per-family informativeness")
    df = pd.read_csv(RES / "bracket_comparison_long.csv")
    df = df[df["suite"] == "synth"].copy()
    df = df.dropna(subset=["K_true"])

    families = ["classic", "noise_sweep", "bg_noise", "varied",
                "high_D", "hierarchical", "adversarial", "imbalanced"]

    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.0),
                             gridspec_kw={"wspace": 0.30, "hspace": 0.50})

    for ax, fam in zip(axes.flat, families):
        sub = df[df["family"] == fam]
        if len(sub) == 0:
            ax.axis("off"); continue

        widths = {
            "MBC": sub["mbc_w"].astype(float).values,
            "HDB": sub["hd_w"].astype(float).values,
            "DBS": sub["db_w"].astype(float).values,
        }
        covers = {
            "MBC": sub["mbc_covers"].astype(float).mean(),
            "HDB": sub["hd_covers"].astype(float).mean(),
            "DBS": sub["db_covers"].astype(float).mean(),
        }
        info = {
            k: covers[k] / (float(np.median(widths[k])) + 1.0)
            for k in widths
        }

        cols = [COL_MBC, COL_HDB, COL_DBS]
        x = np.arange(3)

        ax.bar(x, [info[k] for k in widths.keys()],
               color=cols, alpha=0.88, width=0.66,
               edgecolor="white", linewidth=0.6, zorder=2)

        for xi, k in zip(x, widths.keys()):
            v = info[k]
            if v > 0.12:
                ax.text(xi, v / 2, f"{v:.2f}",
                        ha="center", va="center", fontsize=8.5,
                        fontweight="bold", color="white")
            else:
                ax.text(xi, v + 0.04, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7.5,
                        fontweight="bold", color="0.2")

        ax.set_xticks(x)
        ax.set_xticklabels(list(widths.keys()), fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1.0])
        if fam in ("classic", "high_D"):
            ax.set_ylabel("informativeness", fontsize=8)
        ax.set_title(fam, fontsize=9.5)
        ax.tick_params(axis="y", labelsize=7)
        ax.spines["left"].set_color("0.5")
        ax.spines["bottom"].set_color("0.5")
        ax.grid(axis="y", linestyle=":", alpha=0.35, zorder=0)

    fig.suptitle(r"Per-family bracket informativeness  "
                 r"(coverage $/$ (median width $+\,1$),  "
                 r"$\uparrow$ better)",
                 fontsize=10, y=1.02)
    _save(fig, "fig_per_family_brackets")


# ---------------------------------------------------------------------------
# Fig 3 alt (Option C): 2D calibration scatter
# ---------------------------------------------------------------------------


def fig_calibration_scatter():
    """2D scatter: median bracket width (x) vs coverage (y).
    One point per (family, algorithm). Ideal corner: top-left.
    """
    print("Figure: calibration scatter")
    df = pd.read_csv(RES / "bracket_comparison_long.csv")
    df = df[df["suite"] == "synth"].copy()
    df = df.dropna(subset=["K_true"])

    families = ["classic", "noise_sweep", "bg_noise", "varied",
                "high_D", "hierarchical", "adversarial", "imbalanced"]

    rows = []
    for fam in families:
        sub = df[df["family"] == fam]
        if len(sub) == 0: continue
        rows.append({
            "family": fam,
            "MBC":   (float(np.median(sub["mbc_w"])), sub["mbc_covers"].mean()),
            "HDB":   (float(np.median(sub["hd_w"])),  sub["hd_covers"].mean()),
            "DBS":   (float(np.median(sub["db_w"])),  sub["db_covers"].mean()),
        })

    fig, ax = plt.subplots(figsize=(5.6, 3.4),
                           gridspec_kw={"left": 0.12, "right": 0.74,
                                        "top": 0.89, "bottom": 0.16})

    markers = {"MBC": ("o", COL_MBC, 75),
               "HDB": ("s", COL_HDB, 55),
               "DBS": ("^", COL_DBS, 55)}
    for algo, (m, c, s) in markers.items():
        xs = [r[algo][0] for r in rows]
        ys = [r[algo][1] for r in rows]
        ax.scatter(xs, ys, marker=m, s=s, color=c, alpha=0.85,
                   edgecolor="white", linewidth=0.8, zorder=4,
                   label=f"{algo}{'  (grid)' if algo != 'MBC' else ''}")

    # Family labels at the MBC point (since MBC is the focal algorithm)
    # Manually offset a few that would otherwise overlap.
    label_offsets = {
        "classic":      (0.10, 0.025),
        "high_D":       (0.10, -0.05),
        "imbalanced":   (0.10, 0.025),
        "bg_noise":     (-0.10, -0.07),
        "noise_sweep":  (0.10, 0.025),
        "varied":       (0.10, -0.05),
        "hierarchical": (0.10, 0.03),
        "adversarial":  (0.10, 0.04),
    }
    for r in rows:
        x, y = r["MBC"]
        dx, dy = label_offsets.get(r["family"], (0.15, 0.025))
        ha = "right" if dx < 0 else "left"
        ax.annotate(r["family"], xy=(x, y), xytext=(x + dx, y + dy),
                    ha=ha, va="center", fontsize=7,
                    color=COL_MBC, alpha=0.95)

    ax.set_xlabel(r"median bracket width   ($\leftarrow$ more committal)")
    ax.set_ylabel(r"coverage of $K^\star$   ($\uparrow$ more often correct)")
    ax.set_xlim(-0.6, 9.5)
    ax.set_ylim(-0.05, 1.10)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    ax.scatter([0], [1.0], marker="*", s=220, color="gold",
               edgecolor="0.35", linewidth=0.8, zorder=2, alpha=0.8)
    ax.text(0.18, 1.0, "ideal corner",
            ha="left", va="center", fontsize=7,
            color="0.35", style="italic")

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=False, fontsize=8.5, handletextpad=0.4,
              labelspacing=0.7, title="algorithm",
              title_fontsize=9)
    ax.grid(linestyle=":", alpha=0.35, zorder=0)
    ax.spines["left"].set_color("0.5")
    ax.spines["bottom"].set_color("0.5")

    ax.set_title("Bracket calibration: width vs coverage  "
                 "(one marker per family-algorithm pair)",
                 fontsize=10)
    _save(fig, "fig_calibration_scatter")


# ---------------------------------------------------------------------------
# Fig 3b: per-regime split (appendix candidate)
# ---------------------------------------------------------------------------

def fig_per_regime_brackets():
    """1x3 panels split by MBC's regime classification.
    Tests the regime flag as a structurally-meaningful axis.
    """
    print("Figure: per-regime bracket calibration")
    df = pd.read_csv(RES / "bracket_comparison_long.csv")
    df = df[df["suite"] == "synth"].copy()
    df = df.dropna(subset=["K_true"])

    # Pull per-dataset regime from raw — not in bracket_comparison_long
    raw = pd.read_csv(RES / "synth" / "synth_raw.csv")
    raw_mbc = raw[raw["algo"] == "MBC"]
    regimes = (raw_mbc.groupby("dataset")["regime"]
               .agg(_safe_mode).reset_index())
    df = df.merge(regimes, on="dataset", how="left")

    regimes_order = ["separable", "transitional", "non_separable"]
    regime_titles = {
        "separable":     ("Separable",     "rho > C_high"),
        "transitional":  ("Transitional",  "uncertainty zone"),
        "non_separable": ("Non-separable", "rho < C_low"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.6),
                             gridspec_kw={"wspace": 0.30, "top": 0.80})

    for ax, reg in zip(axes, regimes_order):
        sub = df[df["regime"] == reg]
        if len(sub) == 0:
            ax.axis("off"); continue

        widths = {
            "MBC":     sub["mbc_w"].astype(float).values,
            "HDBSCAN\n(grid)": sub["hd_w"].astype(float).values,
            "DBSCAN\n(grid)":  sub["db_w"].astype(float).values,
        }
        covers = {
            k: (sub[col].astype(float).mean() * 100)
            for k, col in zip(widths.keys(),
                              ["mbc_covers", "hd_covers", "db_covers"])
        }
        cols = [COL_MBC, COL_HDB, COL_DBS]
        positions = [0, 1, 2]

        bp = ax.boxplot(
            [widths[k] for k in widths.keys()],
            positions=positions, widths=0.55,
            patch_artist=True,
            medianprops=dict(color="white", linewidth=1.4),
            whiskerprops=dict(color="0.3", linewidth=0.7),
            capprops=dict(color="0.3", linewidth=0.7),
            flierprops=dict(marker="o", markerfacecolor="0.5",
                            markersize=2.5, markeredgecolor="0.5",
                            linestyle="none"),
            boxprops=dict(linewidth=0.7),
        )
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c); patch.set_alpha(0.85)

        rng = np.random.default_rng(0)
        for pos, (k, c) in zip(positions, zip(widths.keys(), cols)):
            xs = pos + (rng.random(len(widths[k])) - 0.5) * 0.18
            ax.scatter(xs, widths[k], s=10, color=c, alpha=0.6,
                       edgecolor="white", linewidth=0.4, zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels(list(widths.keys()))
        if reg == "separable":
            ax.set_ylabel(r"bracket width $K_{\mathrm{high}} - K_{\mathrm{low}}$")

        ymax = max(np.max(w) for w in widths.values())
        ax.set_ylim(-0.7, max(ymax, 4) * 1.30)
        for pos, k in zip(positions, widths.keys()):
            med = float(np.median(widths[k]))
            ax.text(pos, ymax * 1.04 if ymax > 0 else 1.5,
                    f"med {med:.0f}\ncov {covers[k]:.0f}%",
                    ha="center", va="bottom", fontsize=6.5,
                    color="0.15", linespacing=1.2)

        title, sub_t = regime_titles[reg]
        ax.set_title(f"{title}   ($n_{{\\mathrm{{ds}}}} = {len(sub)}$)",
                     fontsize=9.5, pad=18)
        # Subtitle as in-axes text just below the title
        ax.text(0.5, 1.015, sub_t, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=7.5,
                color="0.35", style="italic")

    fig.suptitle(r"Bracket calibration by MBC's regime classification "
                 r"(synth, $n_{\mathrm{ds}} = 38$)",
                 fontsize=10, y=1.05)
    _save(fig, "fig_per_regime_brackets")


# ---------------------------------------------------------------------------
# Fig 4: three controlled perturbation sweeps on blobs
# ---------------------------------------------------------------------------

def _run_mbc_sweep(loader_fn, strengths, K_true, seed=7, standardize=True):
    """Run MBC on a sequence of perturbation strengths. Returns DataFrame."""
    from MBC import MBCParams, mbc_cluster, standardize_then_mbc
    from sklearn.metrics import adjusted_rand_score

    rows = []
    for s in strengths:
        X, y = loader_fn(s, seed)
        p = MBCParams()
        if standardize:
            res = standardize_then_mbc(X, p, seed=seed)
        else:
            res = mbc_cluster(X, p, seed=seed)
        ari = adjusted_rand_score(y, res.labels) if y is not None else np.nan
        rows.append({
            "strength": s,
            "K_pract": int(res.info.get("K_persistent", res.n_clusters)),
            "K_low":  int(res.bracket[0]),
            "K_high": int(res.bracket[1]),
            "rho":    float(res.rho_hat),
            "regime": res.info.get("regime"),
            "ARI":    ari,
        })
    return pd.DataFrame(rows)


def fig_perturbation_sweep():
    """Three controlled perturbation sweeps on a blobs base.
    Each panel: bracket [K_low, K_high] and K_practical as functions
    of perturbation strength. K_true marked as a horizontal line.
    """
    print("Figure: three perturbation sweeps")
    from sklearn.datasets import make_blobs

    n_base = 1500
    K_true = 4
    SEED = 7

    # ----- Perturbation 1: background contamination -----
    def bg_loader(frac, seed):
        rng = np.random.default_rng(seed)
        n_clean = int(n_base * (1 - frac))
        X, y = make_blobs(n_samples=n_clean, centers=K_true,
                          n_features=2, cluster_std=0.9, random_state=seed)
        if frac > 0:
            lo, hi = X.min(0), X.max(0)
            pad = 0.1 * (hi - lo + 1e-9)
            n_noise = max(1, int(frac * n_base))
            noise = rng.uniform(lo - pad, hi + pad, size=(n_noise, X.shape[1]))
            X = np.vstack([X, noise])
            y = np.concatenate([y, np.full(n_noise, -1)])
        return X, y

    bg_strengths = [0.0, 0.05, 0.10, 0.15, 0.20]
    df_bg = _run_mbc_sweep(bg_loader, bg_strengths, K_true)

    # ----- Perturbation 2: within-cluster scale ratio -----
    # Same K_true=4 blobs, but one cluster has σ scaled by `ratio`.
    # ratio=1 means uniform σ; ratio=8 means one cluster very diffuse.
    def scale_loader(ratio, seed):
        rng = np.random.default_rng(seed)
        per = n_base // K_true
        # Centers on a 6-unit grid
        centers = np.array([[0.0, 0.0], [6.0, 0.0],
                            [0.0, 6.0], [6.0, 6.0]])
        Xs, ys = [], []
        for k in range(K_true):
            sd = 0.7 if k > 0 else 0.7 * ratio
            Xs.append(rng.normal(loc=centers[k], scale=sd, size=(per, 2)))
            ys.append(np.full(per, k))
        return np.vstack(Xs), np.concatenate(ys)

    scale_strengths = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
    df_scale = _run_mbc_sweep(scale_loader, scale_strengths, K_true)

    # ----- Perturbation 3: cluster offset (touching) -----
    # Two clusters in 2D with controlled offset; offset=10 = clearly
    # separable, offset=2 = nearly overlapping.
    def offset_loader(offset, seed):
        rng = np.random.default_rng(seed)
        per = n_base // 2
        c1 = np.array([0.0, 0.0])
        c2 = np.array([offset, 0.0])
        X = np.vstack([
            rng.normal(loc=c1, scale=1.0, size=(per, 2)),
            rng.normal(loc=c2, scale=1.0, size=(per, 2)),
        ])
        y = np.concatenate([np.zeros(per, dtype=int),
                            np.ones(per, dtype=int)])
        return X, y

    offset_strengths = [10.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.0]
    df_off = _run_mbc_sweep(offset_loader, offset_strengths, 2)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8),
                             gridspec_kw={"wspace": 0.30})

    panels = [
        (axes[0], df_bg, "background contamination",
         "fraction of uniform-bg points", K_true),
        (axes[1], df_scale, "within-cluster scale ratio",
         r"$\sigma_{\mathrm{max}} / \sigma_{\mathrm{base}}$", K_true),
        (axes[2], df_off, "cluster offset",
         r"centroid distance $\Delta$  (smaller = harder)", 2),
    ]

    for ax, df, title, xlabel, K_true_panel in panels:
        x = df["strength"].values
        kl = df["K_low"].values
        kh = df["K_high"].values
        kp = df["K_pract"].values

        ax.fill_between(x, kl, kh, color=COL_MBC, alpha=0.2,
                        label=r"bracket $[K_{\mathrm{low}}, K_{\mathrm{high}}]$",
                        zorder=2)
        ax.plot(x, kp, "-o", color=COL_MBC, markersize=5, linewidth=1.6,
                label=r"$\widehat K_{\mathrm{prac}}$", zorder=4)
        ax.plot(x, kh, "--", color=COL_MBC, alpha=0.6, linewidth=0.8,
                zorder=3)
        ax.plot(x, kl, "--", color=COL_MBC, alpha=0.6, linewidth=0.8,
                zorder=3)
        ax.axhline(K_true_panel, color="0.4", linestyle=":", linewidth=1,
                   alpha=0.8, label=r"$K^\star = " + str(K_true_panel) + "$",
                   zorder=2)

        # Annotate regime at each x with colored markers
        for xv, reg in zip(x, df["regime"].values):
            symbol = {"separable": "S", "transitional": "T",
                      "non_separable": "N"}.get(reg, "?")
            color = {"separable": "#27AE60", "transitional": "#E67E22",
                     "non_separable": "#7F8C8D"}.get(reg, "0.5")
            ax.text(xv, -0.07, symbol, ha="center", va="top",
                    fontsize=7, color=color, fontweight="bold",
                    transform=ax.get_xaxis_transform())

        if title.startswith("cluster offset"):
            ax.invert_xaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=9)
        if title.startswith("background"):
            ax.set_ylabel("number of clusters")
            ax.legend(loc="upper left", fontsize=6.5, frameon=False,
                      handletextpad=0.4, labelspacing=0.3)
        ymax = max(int(np.max(kh)) + 1, K_true_panel + 1, 5)
        ax.set_ylim(0.5, ymax)

    # Bottom legend strip — split as 4 evenly-spaced text artists, no overlap
    legend_y = -0.18
    fig.text(0.5, legend_y,
             "Regime markers below x-axis:",
             ha="right", va="top", fontsize=7, color="0.25")
    legend_items = [("S", "#27AE60", "separable"),
                    ("T", "#E67E22", "transitional"),
                    ("N", "#7F8C8D", "non-separable")]
    base_x = 0.52
    spacing = 0.14
    for i, (sym, c, label) in enumerate(legend_items):
        x = base_x + i * spacing
        fig.text(x, legend_y, sym, ha="left", va="top",
                 fontsize=8, color=c, fontweight="bold")
        fig.text(x + 0.012, legend_y, label, ha="left", va="top",
                 fontsize=7, color="0.25")

    fig.suptitle(r"Bracket response to three controlled perturbations  "
                 r"(blobs, $n = 1500$, $K^\star = 4$ except cluster-offset)",
                 fontsize=9.5, y=1.03)
    _save(fig, "fig_perturbation_sweep")


# ---------------------------------------------------------------------------
# Fig 6: K_practical vs K_true scatter with bracket whiskers
# ---------------------------------------------------------------------------

def fig_K_recovery_scatter():
    """K_practical vs K_true scatter, with vertical whiskers showing
    [K_low, K_high]. One point per synth dataset, color by family.
    """
    print("Figure: K-recovery scatter")
    df = pd.read_csv(RES / "synth" / "synth_raw.csv")
    mbc = df[df["algo"] == "MBC"].copy()
    agg = mbc.groupby(["dataset", "family", "K_true"]).agg(
        K_pract=("K", _safe_mode),
        K_low=("bracket_low", "mean"),
        K_high=("bracket_high", "mean"),
        ARI=("ARI", "mean"),
    ).reset_index()
    agg = agg.dropna(subset=["K_true"])

    # Small horizontal jitter when multiple datasets share the same K_true
    rng = np.random.default_rng(0)
    agg["x_jit"] = agg["K_true"] + (rng.random(len(agg)) - 0.5) * 0.30

    # Cap whiskers at a sane height (hier_3x3_2D reaches K_high=38; show 12+ as overflow)
    K_VIS_MAX = 12
    agg["K_high_vis"] = np.minimum(agg["K_high"], K_VIS_MAX)
    agg["overflow"] = agg["K_high"] > K_VIS_MAX

    fig, ax = plt.subplots(figsize=(7.6, 4.6),
                           gridspec_kw={"left": 0.09, "right": 0.78,
                                        "top": 0.92, "bottom": 0.12})

    families_in_order = ["classic", "noise_sweep", "bg_noise", "varied",
                         "high_D", "hierarchical", "adversarial",
                         "imbalanced"]

    K_true_values = sorted(agg["K_true"].unique())

    for fam in families_in_order:
        sub = agg[agg["family"] == fam]
        if len(sub) == 0: continue
        c = FAMILY_COLORS[fam]
        for _, r in sub.iterrows():
            # Whisker
            ax.plot([r["x_jit"], r["x_jit"]],
                    [r["K_low"], r["K_high_vis"]],
                    color=c, alpha=0.5, linewidth=2.6, zorder=2,
                    solid_capstyle="round")
            # Overflow arrow if cap was hit
            if r["overflow"]:
                ax.annotate("", xy=(r["x_jit"], K_VIS_MAX + 0.4),
                            xytext=(r["x_jit"], K_VIS_MAX - 0.2),
                            arrowprops=dict(arrowstyle="-|>", color=c,
                                            lw=0.8, alpha=0.7),
                            zorder=2.5)
        sizes = 25 + 90 * np.clip(sub["ARI"].fillna(0).values, 0, 1) ** 1.5
        ax.scatter(sub["x_jit"], sub["K_pract"], s=sizes, color=c,
                   alpha=0.92, edgecolor="white", linewidth=0.7,
                   zorder=4, label=fam)

    # Diagonal — clipped to visible range
    diag_x = np.linspace(min(K_true_values) - 0.3, max(K_true_values) + 0.3,
                         50)
    ax.plot(diag_x, diag_x, "--", color="0.5", linewidth=0.9,
            zorder=1, alpha=0.7)
    ax.text(max(K_true_values) + 0.1,
            max(K_true_values) + 0.1,
            r"$\widehat K_{\mathrm{prac}} = K^\star$",
            color="0.4", fontsize=7.5, ha="left", va="bottom", style="italic")

    ax.set_xlabel(r"ground-truth number of clusters $K^\star$")
    ax.set_ylabel(r"$\widehat K_{\mathrm{prac}}$ (point) "
                  r"and bracket $[K_{\mathrm{low}}, K_{\mathrm{high}}]$ (whisker)")
    ax.set_xticks(K_true_values)
    ax.set_yticks(range(0, K_VIS_MAX + 1, 2))
    ax.set_xlim(min(K_true_values) - 0.5, max(K_true_values) + 0.7)
    ax.set_ylim(0.3, K_VIS_MAX + 1.0)
    ax.text(min(K_true_values) - 0.4, K_VIS_MAX + 0.6,
            r"$\uparrow$ overflow (bracket extends beyond)",
            fontsize=6.5, color="0.45", style="italic", va="bottom")

    # Family legend OUTSIDE the plot, on the right
    leg = ax.legend(title="family", loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    frameon=False,
                    fontsize=7.5, title_fontsize=8,
                    handletextpad=0.4, labelspacing=0.5)
    leg.get_title().set_fontweight("bold")

    # ARI marker-size legend BELOW the family legend (also outside)
    sz_handles = []
    sz_labels = []
    for a in [0.0, 0.5, 1.0]:
        h = ax.scatter([], [], s=25 + 90 * (a ** 1.5), color="0.4",
                       alpha=0.7, edgecolor="white", linewidth=0.7)
        sz_handles.append(h)
        sz_labels.append(f"ARI = {a:.1f}")
    leg2 = ax.legend(sz_handles, sz_labels, title="point size",
                     loc="lower left", bbox_to_anchor=(1.02, 0.10),
                     frameon=False, fontsize=7.5, title_fontsize=8,
                     handletextpad=0.4, labelspacing=0.6)
    leg2.get_title().set_fontweight("bold")
    ax.add_artist(leg)  # keep both legends

    ax.set_title("K-recovery on 38 synthetic datasets  "
                 "(point on diagonal = perfect recovery, "
                 "short whisker = tight bracket)",
                 fontsize=9.5)
    ax.grid(axis="y", linestyle=":", alpha=0.35, zorder=0)
    _save(fig, "fig_K_recovery_scatter")


# ---------------------------------------------------------------------------
# Retina filtration
# ---------------------------------------------------------------------------

def fig_retina_filtration():
    print("Figure: retina filtration")
    DATA = ROOT / "data"

    Psi = np.load(DATA / "retina_diffmap.npy")
    rgc = np.load(DATA / "rgc_types.npy", allow_pickle=True)
    types = np.array([str(t) for t in rgc])
    type_to_int = {t: i for i, t in enumerate(sorted(np.unique(types)))}
    color_labels = np.array([type_to_int[t] for t in types], dtype=int)

    datasets = ["Retina_full", "Retina_labeled", "V1"]

    def get_data(ds):
        if ds == "Retina_full":
            return Psi, color_labels, "all RGCs"
        if ds == "Retina_labeled":
            mask = color_labels > 0
            return Psi[mask], color_labels[mask], "labeled subset ($K^\\star = 7$)"
        if ds == "V1":
            P = np.load(DATA / "V1-diffmap.npy")
            return P, None, "$K^\\star$ unknown"

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.0),
                             gridspec_kw={"height_ratios": [1.1, 1.0],
                                          "hspace": 0.40, "wspace": 0.32})

    for col, ds in enumerate(datasets):
        info = _load_curve(ds, "neuro")
        X, y, subtitle = get_data(ds)

        ax_top = axes[0, col]
        if y is None:
            ax_top.scatter(X[:, 0], X[:, 1], s=3.5, color="0.35",
                           alpha=0.55, edgecolor="none")
        else:
            uniq = np.unique(y)
            for i, cls in enumerate(uniq):
                m = y == cls
                ax_top.scatter(X[m, 0], X[m, 1], s=3.5,
                               color=PALETTE[i % len(PALETTE)],
                               alpha=0.75, edgecolor="none")
        ax_top.set_xticks([]); ax_top.set_yticks([])
        ds_label = ds.replace("_", "\\_")
        ax_top.set_title(
            f"{ds_label}  ($n = {info['n']}$)\n{subtitle}",
            fontsize=8.6)
        ax_top.text(0.97, 0.03,
                    f"bracket [{info['bracket_low']}, "
                    f"{info['bracket_high']}]\n"
                    f"$\\widehat K_{{\\mathrm{{prac}}}} = "
                    f"{info['K_practical']}$",
                    transform=ax_top.transAxes, ha="right", va="bottom",
                    fontsize=7.5, color=COL_MBC, linespacing=1.3,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec=COL_MBC, alpha=0.9, linewidth=0.6))
        ax_top.set_aspect("equal")
        if col == 0:
            ax_top.set_ylabel("$\\psi_2$  (diffusion-map)")
        ax_top.set_xlabel("$\\psi_1$")

        ax_bot = axes[1, col]
        if info and info["curve"]:
            ks = np.array([p[0] for p in info["curve"]])
            Ks = np.array([p[1] for p in info["curve"]])
            ax_bot.plot(ks, Ks, "-o", color=COL_MBC, markersize=4,
                        linewidth=1.5, zorder=4)
            ax_bot.axvspan(info["k_low"], info["k_high"], alpha=0.15,
                           color=COL_MBC, lw=0)
            ax_bot.axvline(info["k_star"], color=COL_KSTAR, linestyle=":",
                           linewidth=1.2, alpha=0.8)
            ax_bot.axhspan(info["bracket_low"], info["bracket_high"],
                           alpha=0.12, color=COL_MBC, lw=0, zorder=1)
            ax_bot.text(0.97, 0.95,
                        f"regime: {info['regime'].replace('_','-')}",
                        transform=ax_bot.transAxes, ha="right", va="top",
                        fontsize=7.5, color="0.25", style="italic",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  ec="0.7", alpha=0.92, linewidth=0.5))
            ax_bot.set_yscale("log")
            ax_bot.set_ylim(0.7, max(Ks.max(), info["bracket_high"]) * 1.6)
        ax_bot.set_xlabel("$k$")
        if col == 0:
            ax_bot.set_ylabel("$K(k)$  (log)")

    fig.suptitle("Retinal ganglion cells \\& V1: bracket recovers "
                 "neuroscientific estimates without parameter tuning",
                 fontsize=9.5, y=1.00)
    handles = [
        plt.Line2D([0], [0], color=COL_MBC, marker="o", markersize=4,
                   linewidth=1.5, label="$K(k)$"),
        Patch(facecolor=COL_MBC, alpha=0.20,
              label="bracket / uncertainty zone"),
        plt.Line2D([0], [0], color=COL_KSTAR, linestyle=":", linewidth=1.2,
                   label="$k^\\star$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.03), fontsize=7.5)
    _save(fig, "fig_retina_filtration")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

def main():
    fig_per_family_brackets()
    fig_calibration_scatter()
    fig_per_regime_brackets()
    fig_perturbation_sweep()
    fig_K_recovery_scatter()
    fig_retina_filtration()
    print("\nAll figures written to", OUT)


if __name__ == "__main__":
    main()
