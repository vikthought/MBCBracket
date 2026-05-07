#!/usr/bin/env python3
"""Aggregate A_sweep.csv into a tidy per-(dataset, A, candidate_mode)
table by taking the median over the three seeds, plus a small text summary.
"""
from __future__ import annotations
import csv
import statistics as stats
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "ablations" / "A_sweep.csv"
DST = ROOT / "results" / "ablations" / "A_sweep_median.csv"


def med(xs):
    xs = [float(x) for x in xs if x not in ("", "nan")]
    return stats.median(xs) if xs else float("nan")


def maj_mode(xs):
    """Most common string value (ties broken by lexicographic order)."""
    if not xs:
        return ""
    counts = defaultdict(int)
    for x in xs:
        counts[x] += 1
    m = max(counts.values())
    return sorted(k for k, v in counts.items() if v == m)[0]


def main():
    rows = list(csv.DictReader(open(SRC)))
    grouped = defaultdict(list)
    for r in rows:
        key = (r["dataset"], r["regime_class"], int(r["K_true"]),
               float(r["A_coef"]), r["candidate_mode"])
        grouped[key].append(r)

    out = []
    for (name, regime_class, K_true, A, mode), grp in grouped.items():
        out.append({
            "dataset": name,
            "regime_class": regime_class,
            "K_true": K_true,
            "A_coef": A,
            "candidate_mode": mode,
            "k_star_med": int(med([r["k_star"] for r in grp])),
            "K_low_med": int(med([r["K_low"] for r in grp])),
            "K_high_med": int(med([r["K_high"] for r in grp])),
            "K_prac_med": int(med([r["K_prac"] for r in grp])),
            "ARI_med": med([r["ARI"] for r in grp]),
            "rho_med": med([r["rho_hat"] for r in grp]),
            "regime_mode": maj_mode([r["regime"] for r in grp]),
            "n_seeds": len(grp),
        })

    out.sort(key=lambda d: (d["dataset"], d["candidate_mode"], d["A_coef"]))
    fieldnames = list(out[0].keys())
    with open(DST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out:
            w.writerow(r)
    print(f"Wrote {len(out)} median rows to {DST}\n")

    # Pretty-print A-sweep on mutual graph only
    print(f"{'dataset':18s} {'A':>5s} {'k*':>4s} {'K_low':>5s} {'K_high':>6s} "
          f"{'K^':>3s} {'rho':>6s} {'regime':12s} {'ARI':>5s}")
    print("-" * 80)
    for r in out:
        if r["candidate_mode"] != "mutual":
            continue
        print(f"{r['dataset']:18s} {r['A_coef']:5.1f} {r['k_star_med']:4d} "
              f"{r['K_low_med']:5d} {r['K_high_med']:6d} "
              f"{r['K_prac_med']:3d} {r['rho_med']:6.2f} "
              f"{r['regime_mode']:12s} {r['ARI_med']:5.3f}")
    print()
    print("Mutual vs union @ A=1.0:")
    print(f"{'dataset':18s} {'mode':6s} {'K_low':>5s} {'K_high':>6s} "
          f"{'K^':>3s} {'rho':>6s} {'ARI':>5s}")
    print("-" * 60)
    for r in out:
        if r["A_coef"] != 1.0:
            continue
        print(f"{r['dataset']:18s} {r['candidate_mode']:6s} "
              f"{r['K_low_med']:5d} {r['K_high_med']:6d} "
              f"{r['K_prac_med']:3d} {r['rho_med']:6.2f} {r['ARI_med']:5.3f}")


if __name__ == "__main__":
    main()
