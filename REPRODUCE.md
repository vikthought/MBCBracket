# Reproducing the experiments in `PAPER/main.tex`

Every table, figure, and number in the paper is backed by one CSV produced by
one script. This file maps them.

All commands assume you are in the `MBCBracket/` root and have run `./setup.sh`.
Replace `python` with `.venv/bin/python` if your shell does not auto-activate
the venv.

## Headline reproducibility check (≈ 30 s)

```bash
python test_mbc_smoke.py
```

Pins bracket, $\widehat K_{\text{prac}}$, and ARI on three reference datasets at
seed 7. If this passes, your environment reproduces the paper's numbers.

## End-to-end (≈ 1–3 hours, depending on hardware)

```bash
./run_all.sh                            # smoke + synth + neuro + real
python run_real_embedded.py             # §5.3 bracket-vs-grid on embeddings
python analyze_real_embedded.py         # writes results/real_embedded/REPORT.md
python run_perturbation_walk.py         # appendix
python run_sampling_sweep.py            # appendix
python run_sensitivity_grid.py          # appendix
python scripts/run_A_sweep.py           # appendix
python scripts/aggregate_A_sweep.py     # reduces A_sweep.csv → A_sweep_median.csv
python scripts/run_pruning_ablation.py  # appendix
python scripts/gamma_sensitivity.py     # appendix
python scripts/generate_paper_figures.py  # regenerates figures from CSVs
```

## Main-paper map

| paper artifact | section | script | output CSV |
|---|---|---|---|
| Tab. `tab:main-brackets` (synth rows) | §5.1 | `python run_synth.py --seeds 7 11 23` | `results/synth/synth_raw.csv` |
| Tab. `tab:main-brackets` (real rows) | §5.3 | `python run_real.py --seeds 7 11 23` | `results/real/real_raw.csv` |
| Tab. `tab:main-brackets` (neuro rows) | §5.4 | `python run_neuro.py --seeds 7 11 23` | `results/neuro/neuro_raw.csv` |
| Bracket calibration text (§5.2) | §5.2 | (computed from the three CSVs above) | — |
| Tab. `tab:real-bracket-vs-grid` | §5.3 | `python run_real_embedded.py` then `python analyze_real_embedded.py` | `results/real_embedded/real_embedded_raw.csv` (and `REPORT.md`) |
| Fig. `fig:manifold_sampling` | §1 | hand-drawn schematic | `PAPER/figures/FINALNewFigure.jpeg` |
| Fig. `fig:edge_gate` | App. `app:edge-gate-fig` | hand-drawn schematic | `PAPER/figures/RevisedInlineAlgFigure.png` |
| Fig. `fig:per_family_brackets` | App. `app:synth-catalog` | `python scripts/generate_paper_figures.py` | `PAPER/figures/empirical/fig_per_family_brackets.{pdf,png}` |
| Fig. `fig:retina_filtration` | App. `app:neuro-behavior` | `python scripts/generate_paper_figures.py` | `PAPER/figures/empirical/fig_retina_filtration.{pdf,png}` |

> `generate_paper_figures.py` writes into `PAPER/figures/empirical/`. The pre-
> rendered copies actually referenced by `main.tex` live in `PAPER/figures/`
> directly. After regenerating, copy from `empirical/` to `figures/` if you
> want the LaTeX to pick up the fresh versions.

## Appendix map

| paper artifact | appendix | script | output CSV |
|---|---|---|---|
| Tab. `tab:synth-catalog` | App. `app:synth-catalog` | `python run_synth.py` | `results/synth/synth_raw.csv` (seed 7 row) |
| Tab. `tab:per-family` | App. `app:synth-catalog` | (aggregated from synth + baseline grids) | `results/synth/synth_summary.csv` |
| Tab. `tab:A-sweep` | App. `app:A-sweep` | `python scripts/run_A_sweep.py` then `python scripts/aggregate_A_sweep.py` | `results/ablations/A_sweep_median.csv` |
| Tab. `tab:mutual-union` | App. `app:A-sweep` | (subset of `A_sweep.csv` at $A=1$, mutual vs union) | `results/ablations/A_sweep.csv` |
| Tab. `tab:pruning-ablation` | App. `app:A-sweep` | `python scripts/run_pruning_ablation.py` | `results/ablations/pruning_ablation.csv` |
| Tab. `tab:sensitivity` | App. `app:sensitivity` | `python run_sensitivity_grid.py` | `results/sensitivity/sensitivity_grid.csv` |
| Tab. `tab:multi-seed` | App. `app:multi-seed` | (computed from the three main-suite CSVs at seeds 7, 11, 23) | `results/{synth,real,neuro}/*_raw.csv` |
| Tab. `tab:perturbation-walk` | App. `app:perturbation-walk` | `python run_perturbation_walk.py` | `results/perturbations/perturbation_walk.csv` |
| Tab. `tab:sampling-sweep` | App. `app:sampling-sweep` | `python run_sampling_sweep.py` | `results/sampling/sampling_sweep.csv` |
| Tab. `tab:mass-bracket-primary`, `tab:mass-bracket-divergence` | App. `app:mass-bracket` | `python scripts/gamma_sensitivity.py` | `results/gamma_sensitivity.csv` |
| Tab. `tab:real-bracket-detail` | App. `app:real-bracket-detail` | `python run_real_embedded.py` then `python analyze_real_embedded.py` | `results/real_embedded/real_embedded_raw.csv` |

## Inspecting without re-running

The shipped `results/` tree contains every CSV and report listed above. If you
just want to verify a specific number from the paper, open the relevant CSV
directly — for example:

```bash
# bracket / K_prac / ARI on Retina (labeled), seed 7
python -c "import pandas as pd; df=pd.read_csv('results/neuro/neuro_raw.csv'); \
  print(df[(df.dataset=='Retina_labeled') & (df.algo=='MBC') & (df.seed==7)][['bracket_low','bracket_high','K','ARI']])"
```

## Per-runner reference

### Main suites — `run_synth.py`, `run_real.py`, `run_neuro.py`

```text
--seeds 7 11 23           random seeds (one row per (algo, seed))
--out_dir DIR             override results/<suite>/
--datasets PAT [PAT ...]  substring filter on dataset name
--families PAT [PAT ...]  substring filter on family (synth only)
--no_baselines            skip baseline algorithms
--bracket_eval MODE       canonical / low_mid_high / all
--dtm {on,off,both}       DTM rescue mode (default off — no-op on this suite)
```

Extras:

- `run_synth.py --n N` — samples per synthetic dataset (default 2000)
- `run_real.py --max_n N` — subsample real datasets (default 8000)

### Embedded comparison — `run_real_embedded.py`

Estimates each dataset's intrinsic dimension via Levina–Bickel MLE (k=10),
embeds in $\min(\lceil d_{\max}\rceil, D, 32)$ dimensions across PCA, UMAP,
Isomap, and diffusion maps, then runs MBC, DBSCAN-grid (6), HDBSCAN-grid (8) on
each embedding. Output: `results/real_embedded/real_embedded_raw.csv`. Use
`analyze_real_embedded.py` to produce the readable `REPORT.md`.

### Appendix runners

| script | output |
|---|---|
| `run_perturbation_walk.py` | `results/perturbations/perturbation_walk.csv` |
| `run_sampling_sweep.py` | `results/sampling/sampling_sweep.csv` |
| `run_sensitivity_grid.py` | `results/sensitivity/sensitivity_grid.csv` |
| `scripts/run_A_sweep.py` | `results/ablations/A_sweep.csv` |
| `scripts/aggregate_A_sweep.py` | `results/ablations/A_sweep_median.csv` |
| `scripts/run_pruning_ablation.py` | `results/ablations/pruning_ablation.csv` |
| `scripts/gamma_sensitivity.py` | `results/gamma_sensitivity.csv` |

All take few or no flags; default seeds are `7, 11, 23` where applicable.

## Output schema

Each row in `results/<suite>/<suite>_raw.csv`:

- **Metadata**: `dataset, family, D, n, K_true, noise_level, noise_type, seed`
- **Algorithm**: `algo, K_eval_kind` (`main`, `bracket_mid`, `bracket_max`, `bracket_K<N>`, or `baseline`)
- **Metrics**: `K, ARI, NMI, Sil, runtime`
- **MBC-only**: `bracket_low, bracket_high, k_star, k_low, k_high, regime, rho_hat, dtm_rescue_used, n_rescued_edges, d_eff, noise_count, n_edges_final, bracket_K_curve, bracket_Ks_avail`

## Baselines run alongside MBC

Configured in `mbc_runner.run_baselines`:

- **DBSCAN** at defaults plus a 6-point grid (`eps` ∈ {0.7, 1.0, 1.5} × kdist median, `min_samples` ∈ {5, 5·log n})
- **OPTICS, BIRCH** at defaults
- **KMeans, GMM, Ward** (`AgglomerativeClustering(linkage="ward")`), **Spectral** — each at $K=K^\star$ when known and at $K \in \{3, 8\}$ as fallbacks
- **HDBSCAN** at a 4-point `min_cluster_size` grid (0.5, 1, 2, 5 % of $n$), if installed
- Spectral skipped for $n > 3000$ (memory)
