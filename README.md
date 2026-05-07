# MBCBracket

Reproduction bundle for **MBC** ("Manifold-Based Clustering with Persistence
Bracket"). Self-contained: code, cached data and pre-computed results.

The algorithm builds a mutual-$k$NN graph at the empirical logarithmic
connectivity scale, sweeps $k$ across a geometric uncertainty zone derived from
an offset-to-fill proxy, and returns a **bracket** $[K_{\text{low}}, K_{\text{high}}]$
together with a single canonical $K$. The bracket is the headline contribution:
narrow when geometry supports one resolution, wide when several are defensible,
collapsed to $\{1\}$ when no separated structure is detectable.

## Quick start

```bash
./setup.sh                # creates .venv, installs requirements.txt
.venv/bin/python test_mbc_smoke.py    # ~30s: pins bracket / K / ARI on 3 datasets
```

If `test_mbc_smoke.py` passes, the algorithm is working. To reproduce the three
main suites end-to-end (≈ 1–3 hours total):

```bash
./run_all.sh              # smoke + synth + neuro + real, writes results/<suite>/
```

The bundled `results/` directory already contains the outputs from the run that
backs the paper, so you can inspect numbers and figures **without** re-running
anything.

## Directory layout

```text
MBCBracket/
├── README.md                       this file
├── REPRODUCE.md                    paper table/figure → command map
├── requirements.txt                pinned deps (numpy<2 for hdbscan ABI)
├── setup.sh                        creates .venv, installs deps
├── run_all.sh                      smoke + synth + neuro + real
├── check_embedded_run.sh           helper for run_real_embedded.py
│
├── MBC.py                          algorithm — Algorithm 1 of the paper
├── mbc_runner.py                   shared lib (dataset specs, baselines, scoring, plotting)
├── test_mbc_smoke.py               API + regression pin (seed 7)
│
├── run_synth.py                    §5.1 synthetic suite (38 datasets, 8 families)
├── run_real.py                     §5.3 real-world suite (11 datasets)
├── run_neuro.py                    §5.4 retina + V1 (diffusion-map embeddings)
├── run_real_embedded.py            §5.3 bracket-vs-grid on intrinsic-dim embeddings
├── analyze_real_embedded.py        post-processor for the embedded run
├── run_perturbation_walk.py        Tab. perturbation walk (App.)
├── run_sampling_sweep.py           Tab. sampling sweep (App.)
├── run_sensitivity_grid.py         Tab. sensitivity (App.)
│
├── scripts/
│   ├── run_A_sweep.py              Tab. A-sweep (App.)
│   ├── aggregate_A_sweep.py        reduces A_sweep.csv → A_sweep_median.csv
│   ├── run_pruning_ablation.py     Tab. pruning ablation (App.)
│   ├── gamma_sensitivity.py        γ choice for mass-bracket (App.)
│   └── generate_paper_figures.py   regenerates figures from result CSVs
│
├── data/                           cached inputs (≈ 10 MB total)
│   └── README.md                   what's shipped, what's not, how to rebuild
│
├── results/                        pre-computed outputs (≈ 65 MB)
│   ├── synth/{synth_raw.csv, synth_summary.csv, synth_report.md, figs/}
│   ├── real/                       same layout
│   ├── neuro/                      same layout
│   ├── real_embedded/{*.csv, REPORT.md}
│   ├── perturbations/, sampling/, sensitivity/, ablations/
│   └── gamma_sensitivity.csv
│
└── PAPER/
    ├── main.tex, main.pdf, references.bib, neurips_2026.sty
    └── figures/                    pre-rendered figures
```

## Reproducing the paper

[REPRODUCE.md](REPRODUCE.md) maps every table and figure in
the paper to the script that produces the underlying CSV
and the row of `results/` that holds the canonical output.

## Setup notes

`setup.sh` creates `./.venv` (Python 3.10+) and installs everything in
[requirements.txt](requirements.txt). `numpy` is pinned to `<2` because
`hdbscan`'s wheels target the numpy 1.x ABI.

To use a different interpreter:

```bash
PY_BOOT=/opt/homebrew/bin/python3.11 ./setup.sh   # for the venv bootstrap
PY=/usr/local/bin/python3.10 ./run_all.sh         # for the runners
```

## Common run flags

All three main suite drivers (`run_synth.py`, `run_real.py`, `run_neuro.py`)
share these flags via `mbc_runner.add_common_args`:

| flag | default | notes |
|---|---|---|
| `--seeds 7 11 23` | seed `7` only | one row per (algo, seed) |
| `--out_dir DIR` | `results/<suite>/` | override output destination |
| `--datasets PAT...` | all | substring filter on dataset name |
| `--families PAT...` | all | substring filter on family (synth only) |
| `--no_baselines` | off | skip baseline algorithms |
| `--bracket_eval MODE` | `canonical` (synth/real), `mid_max` (neuro) | also score bracket-K partitions: `canonical` / `low_mid_high` / `all` |
| `--dtm {on,off,both}` | `off` | DTM rescue mode (no-op on this suite) |

Per-runner extras:

- `run_synth.py --n N` — samples per synthetic dataset (default 2000)
- `run_real.py --max_n N` — subsample real datasets (default 8000)

## What's not shipped (and why)

- **Raw image data** — `data/MNIST/raw/`, `data/FashionMNIST/raw/`, and
  `data/cifar-10-batches-py/` are excluded (≈ 325 MB). The runner first looks
  for the cached PCA-50 `.npz` files (which **are** shipped) and falls back to
  `sklearn.datasets.fetch_openml` for MNIST. **CIFAR10** has no `.npz`
  fallback — the suite skips it on a clean reproducer. The original CIFAR10
  row remains in `results/real/real_raw.csv` for inspection. See
  [data/README.md](data/README.md) for how to rebuild.
- **Internal notes** — design rationale and audit logs from the development
  process are not part of the bundle; everything load-bearing is in
  [PAPER/main.tex](PAPER/main.tex) and the source comments.

## Verifying reproducibility

`test_mbc_smoke.py` doubles as a regression test: it pins the bracket,
$\widehat K_{\text{prac}}$, and ARI on three reference datasets at seed 7.
If it passes, your environment matches the paper's.
