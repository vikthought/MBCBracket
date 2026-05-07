#!/usr/bin/env python3
"""
analyze_real_embedded.py — read results/real_embedded/real_embedded_raw.csv
and print a markdown report with the comparison tables backing §5.3 and
app:real-bracket-detail of PAPER/main.tex.

Tables produced:
  1. Headline per-dataset table: best embedding for MBC, with bracket / K_hat / ARI.
  2. Full grid: MBC vs DBSCAN-grid vs HDBSCAN-grid across PCA / UMAP / Isomap / Diffusion.
  3. Bracket-coverage and informativeness per (embedding, algo) family.
  4. Per-dataset diagnostic: intrinsic dim estimates and bracket regime.

Usage:
  python analyze_real_embedded.py
  python analyze_real_embedded.py --md results/real_embedded/REPORT.md

Reads:
  results/real_embedded/real_embedded_raw.csv (created by run_real_embedded.py)

Writes:
  stdout (always)
  results/real_embedded/REPORT.md (if --md is given)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

EMB_ORDER = ["pca", "umap", "isomap", "diffusion"]
ALGO_ORDER = ["MBC", "DBSCAN-grid", "HDBSCAN-grid"]
DS_ORDER = [
    "MNIST", "FashionMNIST", "CIFAR10", "Olivetti", "20NG",
    "Digits_pca50", "Pendigits", "Letter",
    "Iris", "Wine", "BreastCancer",
]


def fmt_bracket(lo, hi):
    if pd.isna(lo) or pd.isna(hi):
        return "—"
    return f"[{int(lo)},{int(hi)}]"


def md_table(rows: List[List[str]], header: List[str]) -> str:
    sep = ["---"] * len(header)
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(sep) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def section(title: str) -> List[str]:
    return ["", "## " + title, ""]


def build_report(df: pd.DataFrame) -> str:
    out: List[str] = []
    out.append("# Real-world data with intrinsic-dim embeddings — report")
    out.append("")
    out.append(f"Source: `results/real_embedded/real_embedded_raw.csv` "
               f"({len(df)} rows)")
    out.append(f"Datasets: {df['dataset'].nunique()}, "
               f"embeddings: {sorted(df['embedding'].unique())}, "
               f"seeds: {sorted(df['seed'].unique())}")
    out.append("")

    # Per-dataset 3-seed aggregates
    agg = (df.groupby(["dataset", "embedding", "algo"])
             .agg(n=("n", "first"), D=("D", "first"),
                  K_true=("K_true", "first"),
                  target_dim=("target_dim", "first"),
                  d_max=("d_max", "mean"),
                  d_p95=("d_p95", "mean"),
                  K=("K", "median"),
                  bl=("bracket_low", "median"),
                  bh=("bracket_high", "median"),
                  ARI_mean=("ARI", "mean"),
                  ARI_std=("ARI", "std"))
             .reset_index())

    # ------------------------------------------------------------------
    # 1. Headline: per-dataset best embedding for MBC
    # ------------------------------------------------------------------
    out.extend(section("1. Best MBC embedding per dataset (3-seed mean)"))
    rows = []
    mbc_only = agg[agg["algo"] == "MBC"]
    for ds in DS_ORDER:
        sub = mbc_only[mbc_only["dataset"] == ds]
        if sub.empty:
            continue
        # Highest 3-seed mean ARI (fallback to bracket coverage if all NaN)
        if sub["ARI_mean"].notna().any():
            best = sub.loc[sub["ARI_mean"].idxmax()]
        else:
            best = sub.iloc[0]
        rows.append([
            ds,
            int(best["n"]), int(best["D"]),
            int(best["K_true"]) if not pd.isna(best["K_true"]) else "—",
            int(best["target_dim"]),
            f"{best['d_max']:.1f}",
            best["embedding"],
            fmt_bracket(best["bl"], best["bh"]),
            int(best["K"]),
            f"{best['ARI_mean']:.3f}" if pd.notna(best["ARI_mean"]) else "—",
        ])
    out.append(md_table(rows,
        header=["dataset", "n", "D", "K*", "d_target", "d_max",
                "best embed", "MBC bracket", "K_hat", "MBC ARI"]))

    # ------------------------------------------------------------------
    # 2. Full grid: MBC vs DBSCAN vs HDBSCAN bracket per embedding
    # ------------------------------------------------------------------
    out.extend(section("2. Per (dataset, embedding) — MBC vs DBSCAN/HDBSCAN grids"))
    out.append("Format: `bracket / K_hat / ARI`. ARI is best-of-grid for the "
               "baselines, 3-seed mean across the row.")
    out.append("")
    embeddings_present = [e for e in EMB_ORDER if e in df["embedding"].unique()]
    header = ["dataset", "embed"] + ALGO_ORDER
    rows = []
    for ds in DS_ORDER:
        for emb in embeddings_present:
            sub = agg[(agg["dataset"] == ds) & (agg["embedding"] == emb)]
            if sub.empty:
                continue
            row = [ds, emb]
            for algo in ALGO_ORDER:
                cell = sub[sub["algo"] == algo]
                if cell.empty:
                    row.append("—")
                else:
                    c = cell.iloc[0]
                    ari_str = (f"{c['ARI_mean']:.3f}"
                               if pd.notna(c["ARI_mean"]) else "—")
                    row.append(f"{fmt_bracket(c['bl'], c['bh'])} / "
                               f"{int(c['K'])} / {ari_str}")
            rows.append(row)
    out.append(md_table(rows, header=header))

    # ------------------------------------------------------------------
    # 3. Bracket informativeness per (embedding, algo)
    # ------------------------------------------------------------------
    out.extend(section("3. Bracket informativeness per (embedding, algo)"))
    out.append("Per-seed: coverage = (K_true in bracket), width = bh - bl. "
               "Aggregate: mean coverage / median width / coverage / (med_width + 1).")
    out.append("")
    rows = []
    for emb in embeddings_present:
        for algo in ALGO_ORDER:
            sub = df[(df["embedding"] == emb) & (df["algo"] == algo)].copy()
            if sub.empty:
                continue
            sub = sub[sub["K_true"].notna()]
            if sub.empty:
                continue
            sub["width"] = sub["bracket_high"] - sub["bracket_low"]
            sub["cov"] = ((sub["K_true"] >= sub["bracket_low"]) &
                          (sub["K_true"] <= sub["bracket_high"]))
            cov = sub["cov"].mean()
            mw = sub["width"].median()
            info = cov / (mw + 1)
            rows.append([emb, algo, f"{cov:.2f}", f"{int(mw)}", f"{info:.3f}",
                         f"{sub['ARI'].mean():.3f}"])
    out.append(md_table(rows,
        header=["embed", "algo", "coverage", "median width",
                "informativeness", "mean ARI"]))

    # ------------------------------------------------------------------
    # 4. Diagnostic: intrinsic dim per dataset
    # ------------------------------------------------------------------
    out.extend(section("4. Intrinsic dimension diagnostic"))
    diag = (df.groupby("dataset")
              .agg(D=("D", "first"), n=("n", "first"),
                   d_max=("d_max", "mean"), d_p95=("d_p95", "mean"),
                   target=("target_dim", "first"))
              .reset_index())
    rows = []
    for ds in DS_ORDER:
        if ds not in diag["dataset"].values:
            continue
        r = diag[diag["dataset"] == ds].iloc[0]
        rows.append([ds, int(r["n"]), int(r["D"]),
                     f"{r['d_max']:.1f}", f"{r['d_p95']:.1f}",
                     int(r["target"])])
    out.append(md_table(rows,
        header=["dataset", "n", "D (ambient)", "d_max", "d_p95", "target_dim"]))

    # ------------------------------------------------------------------
    # 5. Best baseline ARI per dataset, all embeddings
    # ------------------------------------------------------------------
    out.extend(section("5. Best baseline ARI per (dataset, embedding)"))
    rows = []
    for ds in DS_ORDER:
        for emb in embeddings_present:
            sub = agg[(agg["dataset"] == ds) & (agg["embedding"] == emb)]
            if sub.empty:
                continue
            mbc = sub[sub["algo"] == "MBC"]
            dbs = sub[sub["algo"] == "DBSCAN-grid"]
            hdb = sub[sub["algo"] == "HDBSCAN-grid"]
            mbc_ari = mbc["ARI_mean"].iloc[0] if not mbc.empty else np.nan
            dbs_ari = dbs["ARI_mean"].iloc[0] if not dbs.empty else np.nan
            hdb_ari = hdb["ARI_mean"].iloc[0] if not hdb.empty else np.nan
            best_label = max(
                [("MBC", mbc_ari), ("DBSCAN-grid", dbs_ari),
                 ("HDBSCAN-grid", hdb_ari)],
                key=lambda t: -1 if pd.isna(t[1]) else t[1]
            )
            rows.append([ds, emb,
                         f"{mbc_ari:.3f}" if pd.notna(mbc_ari) else "—",
                         f"{dbs_ari:.3f}" if pd.notna(dbs_ari) else "—",
                         f"{hdb_ari:.3f}" if pd.notna(hdb_ari) else "—",
                         best_label[0]])
    out.append(md_table(rows,
        header=["dataset", "embed", "MBC", "DBS", "HDB", "best"]))

    # ------------------------------------------------------------------
    # 6. The takeaway
    # ------------------------------------------------------------------
    out.extend(section("6. Summary statistics"))
    overall = []
    for emb in embeddings_present:
        for algo in ALGO_ORDER:
            sub = df[(df["embedding"] == emb) & (df["algo"] == algo)]
            if sub.empty:
                continue
            overall.append([emb, algo,
                            f"{sub['ARI'].mean():.3f}",
                            f"{sub['ARI'].std():.3f}",
                            int(len(sub))])
    out.append(md_table(overall,
        header=["embed", "algo", "mean ARI", "ARI std", "n_rows"]))

    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv",
                    default="results/real_embedded/real_embedded_raw.csv")
    ap.add_argument("--md",
                    default="results/real_embedded/REPORT.md",
                    help="Where to also write the report (default writes one).")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}\n"
                 "Run `python run_real_embedded.py` first.")
    df = pd.read_csv(csv_path)
    if df.empty:
        sys.exit("CSV is empty.")

    report = build_report(df)
    print(report)

    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text(report)
        print(f"\n[wrote {args.md}]")


if __name__ == "__main__":
    main()
