# Cached datasets

What's shipped here is exactly what the runners need to reproduce the paper's
real-world and neuroscience tables — about 10 MB total. The full raw bundles
(MNIST, FashionMNIST, CIFAR-10) are intentionally excluded because they are
either fetchable on demand or large and out of scope.

## Shipped files

| file | size | source | used by |
|---|---|---|---|
| `retina_diffmap.npy`, `rgc_types.npy` | 130 KB | retinal-ganglion-cell recording, diffusion-map embedded | `run_neuro.py` (Retina_full / labeled / off_brisk / non_off_brisk) |
| `V1-diffmap.npy` | 30 KB | primary-visual-cortex recording, diffusion-map embedded | `run_neuro.py` (V1) |
| `Digits_pca50.npz` | 332 KB | UCI handwritten digits → PCA-50 | `run_real.py` |
| `Olivetti_pca50.npz` | 156 KB | sklearn Olivetti faces → PCA-50 | `run_real.py` |
| `Pendigits.npz` | 244 KB | UCI Pen-Based Recognition of Handwritten Digits | `run_real.py` |
| `Letter.npz` | 248 KB | UCI Letter Recognition | `run_real.py` |
| `MNIST_pca50_n8000.npz` | 1.4 MB | MNIST (8000 sample) → PCA-50 | `run_real.py` |
| `FashionMNIST_pca50_n8000.npz` | 1.4 MB | FashionMNIST (8000 sample) → PCA-50 | `run_real.py` |
| `20NG_svd100_n8000.npz` | 5.9 MB | 20 Newsgroups (8000 sample) → TF-IDF + SVD-100 | `run_real.py` |

The retinal data are diffusion-map embeddings from
[Dyballa & Zucker, *Functional connectivity guides clustering of retinal
ganglion cells*](https://doi.org/10.1101/2024.02.20.581293).

## What is **not** shipped

| not shipped | size | how the runner reacts |
|---|---|---|
| `data/MNIST/raw/` | ~63 MB | Not needed — `MNIST_pca50_n8000.npz` is the actual input. The loader falls back to `sklearn.datasets.fetch_openml('mnist_784')` if the cache is ever missing. |
| `data/FashionMNIST/raw/` | ~82 MB | Not needed — `FashionMNIST_pca50_n8000.npz` is the actual input. No `fetch_openml` fallback for this one. |
| `data/cifar-10-batches-py/` | ~178 MB | **CIFAR-10 will be skipped.** No `.npz` cache exists; the loader returns `None` and the runner just prints a skip message. The original CIFAR-10 row remains in `results/real/real_raw.csv` for inspection. To rerun: download `cifar-10-batches-py` from <https://www.cs.toronto.edu/~kriz/cifar.html> and place it here. |

## Rebuilding caches from raw

If you want bit-exact rebuilds of the `_pca50` / `_svd100` caches:

```python
# MNIST_pca50_n8000.npz — produced inside _load_mnist_with_fetch in mbc_runner.py
# (the runner does this automatically the first time it can't find the cache).

# FashionMNIST_pca50_n8000.npz / 20NG_svd100_n8000.npz — produced by separate
# preprocessing scripts not part of this bundle. The shipped caches are the
# canonical inputs; rebuild them only if you need to vary the seed or PCA dim.
```

The PCA-50 / SVD-100 representations are the actual inputs the paper's
`tab:main-brackets` and §5.3 are computed on. Reproducing the paper does not
require regenerating them.
