#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MBC: Manifold-Based Clustering with Persistence Bracket
=======================================================

Implements Algorithm 1 from PAPER/main.tex with three theoretically-grounded
choices:

  - **Per-edge gate (Theorem 1's no-bridge bound).** Mutual-kNN candidate
    edges are filtered by ||x_i - x_j|| <= alpha * min(D_k(x_i), D_k(x_j)).
    This is exactly the per-edge condition used in the proof: any
    same-component mutual-kNN edge automatically satisfies it, while
    cross-component pairs in the no-bridge regime have ||x_i - x_j|| >=
    Delta > C * h_max >= both individual radii and so violate it.
    alpha = 1.5 by default (small slack for the transitional regime).

  - **Cluster size threshold = max(ceil(0.005 * |A|), k_star, 5).** The
    0.005 fraction is the paper-canonical value (main.tex line 237). The
    k_star floor is motivated by Theorem 1's cap-occupancy argument,
    which requires components to have Omega(log n) samples for the
    radius-concentration bound to apply uniformly. Components below this
    floor are not certified by the theorem and are treated as dust. The
    absolute 5 protects very small datasets where the fractional bound
    rounds below 5.

  - **K-selection: paper-strict primary, practical fall-through.**
    K_persistent_strict = #{components alive throughout [k_low, k_high]}
    is the paper's exact definition (Eq. 6 in main.tex). For abrupt-merge
    data (e.g. retina_full, where every non-giant component dies within
    one sweep step) this collapses to K=1, which is the conservative
    answer the paper formalizes. K_practical falls through to the mode
    of K_big(k) over the zone, restricted to K > 1, when K_strict <= 1.
    Both K's are exposed in MBCResult.info; n_clusters reflects K_practical.

Public API:
  - MBCParams
  - MBCResult            (with K_persistent_strict in info)
  - mbc_cluster(X, params, seed=0) -> MBCResult
  - mbc_cluster_legacy(X, params, seed=0) -> (labels, info_dict)
  - standardize_then_mbc(X, params=None, seed=0) -> MBCResult
  - get_labels_for_K(res, K_target) -> Optional[np.ndarray]
  - summarize(res) -> str
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, List, Optional, Iterable

import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Utilities
# =============================================================================

def _connected_components_from_edges(n: int, edges: np.ndarray) -> np.ndarray:
    """Union-find connected components. edges: (m,2) int array."""
    parent = np.arange(n, dtype=int)
    size = np.ones(n, dtype=int)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    if edges.size:
        for i, j in edges:
            union(int(i), int(j))

    for i in range(n):
        parent[i] = find(i)

    _, labels = np.unique(parent, return_inverse=True)
    return labels


def _effective_dimensionality(X: np.ndarray, var_thresh: float = 0.90,
                              max_comp: int = 64) -> int:
    """Smallest #PCs explaining >= var_thresh of variance, capped at max_comp."""
    n, D = X.shape
    if D <= 1 or n <= 3:
        return max(1, D)
    m = min(D, max_comp, max(2, min(D, n - 1)))
    try:
        pca = PCA(n_components=m, svd_solver="randomized", random_state=0)
        evr = pca.fit(X).explained_variance_ratio_
        d_eff = int(np.searchsorted(np.cumsum(evr), var_thresh) + 1)
        return max(1, min(d_eff, D))
    except Exception:
        return max(1, min(D, m))


def _fit_nn(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (dists, inds) including self at column 0."""
    n, D = X.shape
    k = int(max(1, min(k, n - 1)))
    algo = "brute" if D >= 30 else "auto"
    nn = NearestNeighbors(n_neighbors=min(k + 1, n), algorithm=algo,
                          metric="euclidean").fit(X)
    return nn.kneighbors(X, return_distance=True)


def _pairs_to_array(pairs: Iterable[tuple]) -> np.ndarray:
    P = list(pairs)
    if not P:
        return np.zeros((0, 2), dtype=int)
    A = np.fromiter((x for t in P for x in t), dtype=int, count=2 * len(P))
    return A.reshape(-1, 2)


def _fit_nn_mreach(X: np.ndarray, k: int, cores: np.ndarray,
                   over_factor: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Top-k neighbors under mutual-reachability distance.

    d_mreach(i, j) = max(core_i, core_j, ||x_i - x_j||).

    To avoid O(n^2) work we first take ``over_factor * k`` Euclidean
    nearest neighbors for each point, transform their distances to
    d_mreach, and re-rank. With ``over_factor >= 2`` the Euclidean window
    is comfortably larger than the d_mreach top-k for the high-D regimes
    we care about; pathological exceptions where a true d_mreach top-k
    neighbor sits outside the Euclidean window are negligible at
    operating ``k`` values.
    """
    n, D = X.shape
    k = int(max(1, min(k, n - 1)))
    n_get = int(min(over_factor * (k + 1), n))
    algo = "brute" if D >= 30 else "auto"
    nn = NearestNeighbors(n_neighbors=n_get, algorithm=algo,
                          metric="euclidean").fit(X)
    dists_e, inds = nn.kneighbors(X, return_distance=True)
    # Pairwise core max: core_i vs core_j for j in inds[i]
    core_pair = np.maximum(cores[:, None], cores[inds])
    dists_mr = np.maximum(dists_e, core_pair)
    # Re-sort each row by mutual-reachability distance
    order = np.argsort(dists_mr, axis=1, kind="stable")
    inds_sorted = np.take_along_axis(inds, order, axis=1)
    dists_sorted = np.take_along_axis(dists_mr, order, axis=1)
    return dists_sorted[:, : k + 1], inds_sorted[:, : k + 1]


# =============================================================================
# Parameters
# =============================================================================

@dataclass
class MBCParams:
    """Parameters for MBC.

    Defaults are tuned for the new pipeline. Old tuning-knob names are
    preserved for backward compat, even when no longer used in the main path.
    """

    # ---- core scale parameters ----
    delta: float = 0.05
    A_coef: float = 1.0

    # ---- candidate levers (default OFF, preserve baseline behavior) ----
    # Lever 1: cap d_eff used in threshold-constant inversion. PCA at 90%
    # EVR overestimates intrinsic manifold dimension on noisy high-D data;
    # capping at the manifold-hypothesis prior dim widens the inverted
    # uncertainty zone on real-world high-D data only.
    d_eff_cap: int = 64

    # Lever 2: replace Euclidean kNN with mutual-reachability kNN
    # (d_mreach(i,j) = max(core_i, core_j, ||x_i-x_j||)). Density-aware
    # metric used in HDBSCAN; partially breaks distance concentration in
    # high D by lifting all sub-core distances to the core scale.
    use_mutual_reachability: bool = False

    # ---- candidate graph ----
    # "mutual" (default, new):  edges {i,j} with i in N_k(j) AND j in N_k(i)
    # "union"  (legacy):        edges {i,j} with i in N_k(j) OR  j in N_k(i)
    candidate_mode: str = "mutual"

    # When mutual-kNN leaves a node isolated, fall back to its single nearest
    # neighbor so it joins something. Used in the LABEL graph at k* only;
    # the persistence sweep does NOT use the fallback (it would break
    # filtration monotonicity).
    mutual_isolated_fallback: bool = True
    # If True, also use the fallback inside the persistence sweep. Default
    # False because the fallback adds non-mutual edges that can disappear
    # at larger k as those nodes acquire mutual neighbors, breaking the
    # filtration property. Kept as an ablation knob.
    persistence_use_fallback: bool = False

    # ---- per-edge threshold gate ----
    # The condition  ||x_i - x_j|| <= alpha * min(D_k(x_i), D_k(x_j))  is
    # automatically satisfied by every reciprocal mutual-kNN edge with
    # alpha >= 1, because reciprocity forces ||x_i - x_j|| <= D_k(x_i) AND
    # ||x_i - x_j|| <= D_k(x_j) separately. The gate therefore acts only
    # on the NON-RECIPROCAL fallback edges added when isolated nodes are
    # connected to their nearest neighbor: it filters fallback edges that
    # would cross a density discontinuity (point i in a sparse region,
    # nearest neighbor j in a dense one). We keep the gate ON in the
    # main / label graph (where the fallback runs) and force it OFF in
    # the persistence sweep (which has no fallback to filter, so the
    # gate would be a no-op anyway). alpha = 1.5 admits the typical
    # (1 +/- epsilon) deviation in the radius-concentration bound.
    use_min_h_gate: bool = True
    min_h_gate_alpha: float = 1.5

    # ---- legacy gate / triangle (default OFF) ----
    # Kept for backward-compat ablations.
    use_geometric_mean_gate: bool = False
    gm_gate_alpha: float = 1.5
    tri_min_shared: int = 0

    # ---- local-k schedule ----
    use_local_k: bool = True
    enforce_k_floor: bool = True
    # 0.5 floors k_low at ~0.5·log(4n/δ); for retina (log_term≈11.4) that's 6,
    # which sits one step past the merge point — the bracket-high tops out at
    # K=3 because the structure at k=5 is invisible. 0.4 lowers the floor to
    # k=5 for retina/V1-scale data, while keeping it well above the
    # k=2 dust-fragmentation regime. Pairs with the bracket size floor below.
    kmin_log_multiplier: float = 0.4
    kmax_factor: float = 4.0

    # ---- density pruning (NEW; default ON) ----
    # Prune points whose pilot kNN radius exceeds q_noise quantile of the
    # radius distribution BEFORE building the connectivity graph. DBSCAN-style
    # noise demotion that makes Theorem 1 robust to background contamination
    # without changing its hypotheses.
    use_density_pruning: bool = True
    q_noise: float = 0.95
    noise_factor: float = 1.5
    noise_reassign: bool = True
    noise_reassign_factor: float = 2.0

    # bracket_labels exposes labels for each K in the bracket. If True
    # (default), every point is force-assigned to its nearest cluster so the
    # array contains exactly K unique values (no -1 noise). The noise-aware
    # version of these labels is always available under
    # MBCResult.bracket_labels_with_noise.
    bracket_labels_force_assign: bool = True

    # ---- DTM rescue (kept for compat; default OFF) ----
    use_dtm_rescue: bool = False
    trim_mult: float = 4.0
    max_within_eval: int = 32
    q_high: float = 0.90
    m_frac: float = 0.5

    # ---- persistence bracket ----
    epsilon_no_bridge: float = 0.5
    a_collar: float = 0.05
    use_data_driven_bracket: bool = True
    A_factor_fallback: float = 1.5
    # Quantile of the active-subset radius distribution used as a proxy
    # for h_max in the rho-hat estimator. 0.5 (median) is the original
    # default; higher quantiles (0.9, 0.95) give more conservative
    # (smaller) rho-hat in line with the worst-case fill distance the
    # threshold theorem analyzes.
    rho_h_quantile: float = 0.5
    # Sweep range (multipliers on A_coef). New defaults reach further DOWN
    # than before, since several failure modes show K_true at k < k*.
    bracket_A_min_ratio: float = 0.15
    bracket_A_max_ratio: float = 4.0

    bracket_n_sweep: int = 15

    # ---- minimum cluster size for K-selection (PAPER CANONICAL) ----
    # PAPER/main.tex line 237 fixes "a minimum-component fraction 0.005".
    # We use that value here so the algorithm matches what the paper claims.
    # An absolute floor of 5 points is applied as a small-n safeguard
    # (datasets with n<1000 round below 5 under 0.5%, where any "cluster" of
    # 1-4 points is dust).
    cluster_min_size_frac: float = 0.005

    # ---- mass-coverage companion bracket ----
    # In addition to the headline (size-filtered) bracket, MBC reports a
    # mass-coverage bracket: at each scale k, K_mass(k) is the smallest m
    # such that the m largest active components cumulatively cover at least
    # ``mass_gamma`` of the active mass. The bracket is the (min, max)
    # range of K_mass(k) over the zone. ``bracket_mass_persistent``
    # additionally requires K_mass(k) to hit a value for at least
    # ``mass_min_run`` consecutive k's, dropping single-step flickers.
    #
    # Companion, not headline. The size-filtered bracket is what Theorem 1
    # directly licenses (cap-occupancy: components below s_min are not
    # certified). The mass bracket is a relative-size diagnostic that can
    # admit long-tail minorities the absolute filter drops, at the cost of
    # exploding on uniform-mass fragmentation in high-D Euclidean data. See
    # PAPER appendix sensitivity study.
    mass_gamma: float = 0.95
    mass_min_run: int = 2


# =============================================================================
# Result
# =============================================================================

@dataclass
class MBCResult:
    labels: np.ndarray
    n_clusters: int
    bracket: Tuple[int, int]
    bracket_K_curve: List[Tuple[int, int]]
    bracket_labels: Dict[int, np.ndarray]
    bracket_k_for_K: Dict[int, int]
    bracket_sweep: List[Dict[str, Any]]
    k_star: int
    k_low: int
    k_high: int
    rho_hat: float
    rho_hat_lower: float
    C_lower: float
    C_upper: float
    A_low: float
    A_high: float
    d_eff: int
    n_edges_final: int
    n_edges_remove_only: int
    dtm_rescue_used: bool
    n_rescued_edges: int
    noise_count: int
    H_median: float
    H_min: float
    H_max: float
    info: Dict[str, Any] = field(default_factory=dict)
    bracket_labels_with_noise: Dict[int, np.ndarray] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "labels": self.labels.tolist(),
            "n_clusters": self.n_clusters,
            "bracket": list(self.bracket),
            "bracket_K_curve": [list(t) for t in self.bracket_K_curve],
            "bracket_labels": {int(K): L.tolist()
                               for K, L in self.bracket_labels.items()},
            "bracket_labels_with_noise": {int(K): L.tolist()
                                          for K, L in self.bracket_labels_with_noise.items()},
            "bracket_k_for_K": {int(K): int(k)
                                for K, k in self.bracket_k_for_K.items()},
            "bracket_sweep": [
                {"k": int(s["k"]), "K": int(s["K"]),
                 "labels": s["labels"].tolist(),
                 "n_edges": int(s["n_edges"])}
                for s in self.bracket_sweep
            ],
            "k_star": self.k_star, "k_low": self.k_low, "k_high": self.k_high,
            "rho_hat": float(self.rho_hat),
            "rho_hat_lower": float(self.rho_hat_lower),
            "C_lower": float(self.C_lower), "C_upper": float(self.C_upper),
            "A_low": float(self.A_low), "A_high": float(self.A_high),
            "d_eff": int(self.d_eff),
            "n_edges_final": int(self.n_edges_final),
            "n_edges_remove_only": int(self.n_edges_remove_only),
            "dtm_rescue_used": bool(self.dtm_rescue_used),
            "n_rescued_edges": int(self.n_rescued_edges),
            "noise_count": int(self.noise_count),
            "H_median": float(self.H_median),
            "H_min": float(self.H_min), "H_max": float(self.H_max),
            "info": self.info,
        }


# =============================================================================
# Pilot scale
# =============================================================================

def _choose_k_star(n: int, delta: float, A_coef: float) -> int:
    log_term = math.log(max(3.0, 4.0 * n) / max(1e-12, float(delta)))
    k_star = int(math.ceil(A_coef * log_term))
    return min(max(2, k_star), max(2, n - 1))


def _local_k_schedule(H_pilot: np.ndarray, d_eff: int, k_star: int,
                      kmin: int, kmax: int, enforce_floor: bool) -> np.ndarray:
    H = np.asarray(H_pilot, float)
    Href = float(np.median(H[H > 0])) if np.any(H > 0) else float(np.median(H))
    H_safe = np.maximum(H, 1e-12)
    ratio = Href / H_safe

    d_use = max(1, int(d_eff))
    target_max = max(1.0, (kmax * 100.0) / max(1.0, float(k_star)))
    ratio_max = target_max ** (1.0 / d_use)
    ratio_min = 1.0 / max(ratio_max, 1.0)
    ratio_clamped = np.clip(ratio, ratio_min, ratio_max)

    base = k_star * (ratio_clamped ** d_use)
    base = np.nan_to_num(base, nan=float(k_star),
                         posinf=float(kmax), neginf=float(kmin))
    k_i = np.floor(base).astype(int)
    k_i = np.clip(k_i, kmin, kmax)
    if enforce_floor:
        k_i = np.maximum(k_i, int(max(1, k_star)))
    return k_i


# =============================================================================
# Threshold constants
# =============================================================================

def _threshold_constants(A: float, d: int, eps: float, a: float,
                         c_ratio: float = 1.0, R: float = 1.0
                         ) -> Tuple[float, float]:
    B = 1.0 + 2.0 * a
    one_minus_eps = max(1e-9, 1.0 - eps)
    over = 2.0 / (one_minus_eps ** (1.0 / d)) * (2.0 * A * R * c_ratio) ** (1.0 / d)
    under = (A * R / (2.0 ** (d + 2) * c_ratio * (B ** d))) ** (1.0 / d)
    return float(under), float(over)


# =============================================================================
# Density pruning
# =============================================================================

def _density_prune(H_pilot: np.ndarray, p: MBCParams
                   ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Identify points to prune as 'too sparse' (background noise).

    Returns (active_mask, pruned_mask, tau_noise).
    """
    n = len(H_pilot)
    if not p.use_density_pruning or p.q_noise >= 1.0:
        return np.ones(n, dtype=bool), np.zeros(n, dtype=bool), float("inf")

    H_pos = H_pilot[H_pilot > 0]
    if H_pos.size == 0:
        return np.ones(n, dtype=bool), np.zeros(n, dtype=bool), float("inf")

    q_val = float(np.quantile(H_pos, p.q_noise))
    tau = q_val * float(p.noise_factor)
    pruned = H_pilot > tau
    active = ~pruned

    # Safety: never prune more than 50% of the data
    if pruned.sum() > 0.5 * n:
        tau = float(np.quantile(H_pos, 0.5))
        pruned = H_pilot > tau
        active = ~pruned

    return active, pruned, tau


def _reassign_pruned(X: np.ndarray, full_labels_in: np.ndarray,
                     active_mask: np.ndarray, pruned_mask: np.ndarray,
                     H_pilot: np.ndarray, p: MBCParams,
                     ind_full: np.ndarray) -> np.ndarray:
    """Reassign pruned points by majority vote among nearby active neighbors.

    full_labels_in: (n,) full-array labels with -1 for pruned. Returns a copy
    with pruned points reassigned where possible (still -1 otherwise).
    """
    full_labels = full_labels_in.copy()

    if not p.noise_reassign or pruned_mask.sum() == 0:
        return full_labels

    H_active_pos = H_pilot[active_mask & (H_pilot > 0)]
    H_ref = float(np.median(H_active_pos)) if H_active_pos.size \
        else float(np.median(H_pilot[H_pilot > 0]))
    H_ref = max(H_ref, 1e-12)
    radius_thr = p.noise_reassign_factor * H_ref

    pruned_idx = np.where(pruned_mask)[0]
    for i in pruned_idx:
        nbrs = ind_full[i, 1:]
        nbr_d = np.linalg.norm(X[nbrs] - X[i], axis=1)
        is_active = active_mask[nbrs]
        is_close = nbr_d <= radius_thr
        ok = is_active & is_close
        if not np.any(ok):
            continue
        cand_labels = full_labels[nbrs[ok]]
        cand_labels = cand_labels[cand_labels >= 0]
        if cand_labels.size == 0:
            continue
        vals, counts = np.unique(cand_labels, return_counts=True)
        full_labels[i] = int(vals[np.argmax(counts)])

    return full_labels


def _force_assign_all(X: np.ndarray, labels_in: np.ndarray,
                      ind_full: np.ndarray, K: int) -> np.ndarray:
    """Force every -1 point in labels_in to be assigned to one of K clusters.

    For each -1 point: walk its (precomputed) nearest-neighbor list and take
    the first non-(-1) label encountered. If every neighbor is also -1
    (extremely rare), fall back to brute-force over assigned points. The
    result contains labels in {0, ..., K-1} and no -1.
    """
    out = labels_in.copy()
    noise_idx = np.where(out == -1)[0]
    if noise_idx.size == 0:
        return out

    needs_brute: List[int] = []
    for i in noise_idx:
        nbrs = ind_full[i, 1:]
        labs = out[nbrs]
        ok = labs >= 0
        if np.any(ok):
            j = int(nbrs[np.argmax(ok)])  # first True is nearest
            out[i] = int(out[j])
        else:
            needs_brute.append(int(i))

    if needs_brute:
        assigned_idx = np.where(out >= 0)[0]
        if assigned_idx.size == 0:
            # Degenerate: every point is -1. Put everything in cluster 0.
            return np.zeros_like(out)
        for i in needs_brute:
            d = np.linalg.norm(X[assigned_idx] - X[i], axis=1)
            j = int(assigned_idx[np.argmin(d)])
            out[i] = int(out[j])

    # If unique labels < K (some kept cluster ended up empty), relabel to
    # contiguous 0..M-1; the caller asked for K but data only supports M.
    unique = np.unique(out)
    if unique.size != K:
        remap = {int(old): new for new, old in enumerate(unique)}
        out = np.array([remap[int(v)] for v in out], dtype=out.dtype)
    return out


# =============================================================================
# Edge construction
# =============================================================================

def _build_kNN_lists(inds: np.ndarray, k_i: np.ndarray) -> List[np.ndarray]:
    n = len(k_i)
    out: List[np.ndarray] = []
    maxk = inds.shape[1] - 1
    for i in range(n):
        ki = int(min(max(1, k_i[i]), maxk))
        out.append(inds[i, 1:ki + 1].astype(int))
    return out


def _mutual_pairs(lists: List[np.ndarray]) -> np.ndarray:
    sets = [set(lst.tolist()) for lst in lists]
    pairs = set()
    for i, lst in enumerate(lists):
        for j in lst:
            j = int(j)
            if j <= i:
                continue
            if i in sets[j]:
                pairs.add((i, j))
    return _pairs_to_array(pairs)


def _union_pairs(lists: List[np.ndarray]) -> np.ndarray:
    pairs = set()
    for i, lst in enumerate(lists):
        for j in lst:
            j = int(j)
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            pairs.add((a, b))
    return _pairs_to_array(pairs)


def _add_isolation_fallback(edges: np.ndarray, n: int,
                            ind_full: np.ndarray) -> np.ndarray:
    """For any node with degree 0, add an edge to its nearest neighbor."""
    if edges.size:
        deg = np.zeros(n, dtype=int)
        for (i, j) in edges:
            deg[i] += 1
            deg[j] += 1
    else:
        deg = np.zeros(n, dtype=int)

    extra = []
    for i in range(n):
        if deg[i] == 0:
            j = int(ind_full[i, 1]) if ind_full.shape[1] >= 2 else -1
            if j >= 0 and j != i:
                a, b = (i, j) if i < j else (j, i)
                extra.append((a, b))
    if not extra:
        return edges
    if edges.size == 0:
        return _pairs_to_array(set(extra))
    all_edges = set(map(tuple, edges.tolist())) | set(extra)
    return _pairs_to_array(all_edges)


def _build_graph(X: np.ndarray, k_i_full: np.ndarray, p: MBCParams,
                 active_idx: np.ndarray,
                 cores_full: Optional[np.ndarray] = None
                 ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """Build connectivity graph on the active subset.

    Returns (edges_active_local, lists_active_local, H_active).
    If ``p.use_mutual_reachability`` is True and ``cores_full`` is
    provided, the kNN graph is built under the mutual-reachability
    distance d_mreach(i,j) = max(core_i, core_j, ||x_i - x_j||) instead
    of plain Euclidean.
    """
    n_active = active_idx.size
    if n_active < 2:
        return (np.zeros((0, 2), dtype=int),
                [np.zeros(0, dtype=int)] * n_active,
                np.zeros(n_active, dtype=float))

    X_act = X[active_idx]
    k_max = int(min(n_active - 1, max(int(k_i_full.max()), 2)))
    if p.use_mutual_reachability and cores_full is not None:
        cores_act = cores_full[active_idx]
        d_act, ind_act = _fit_nn_mreach(X_act, k_max, cores_act)
    else:
        d_act, ind_act = _fit_nn(X_act, k_max)

    H_act = np.empty(n_active, dtype=float)
    maxcol = ind_act.shape[1] - 1
    k_act = np.empty(n_active, dtype=int)
    for ai, fi in enumerate(active_idx):
        kk = int(min(max(1, k_i_full[fi]), maxcol))
        H_act[ai] = d_act[ai, kk]
        k_act[ai] = kk

    lists_act = _build_kNN_lists(ind_act, k_act)

    if p.candidate_mode == "mutual":
        edges = _mutual_pairs(lists_act)
        if p.mutual_isolated_fallback:
            edges = _add_isolation_fallback(edges, n_active, ind_act)
    else:
        edges = _union_pairs(lists_act)

    # Per-edge threshold gate: ||x_i - x_j|| <= alpha * min(H_i, H_j).
    # This is the theorem's per-edge condition. In the separable regime it's
    # a no-op (all mutual-kNN edges already satisfy it); in the transitional
    # regime it filters bridge points whose paired-with-a-deep-cluster-member
    # endpoint has a large H_i.
    if p.use_min_h_gate and edges.size:
        dij = np.linalg.norm(X_act[edges[:, 0]] - X_act[edges[:, 1]], axis=1)
        thr = float(p.min_h_gate_alpha) * np.minimum(
            H_act[edges[:, 0]], H_act[edges[:, 1]]
        )
        edges = edges[dij <= thr]

    # Optional legacy gate (default off in new path)
    if p.use_geometric_mean_gate and edges.size:
        dij = np.linalg.norm(X_act[edges[:, 0]] - X_act[edges[:, 1]], axis=1)
        thr = float(p.gm_gate_alpha) * np.sqrt(H_act[edges[:, 0]] * H_act[edges[:, 1]])
        edges = edges[dij <= thr]

    if p.tri_min_shared >= 1 and edges.size:
        sets_act = [set(lst.tolist()) for lst in lists_act]
        keep = np.zeros(len(edges), dtype=bool)
        for t, (i, j) in enumerate(edges):
            keep[t] = len(sets_act[int(i)] & sets_act[int(j)]) >= int(p.tri_min_shared)
        edges = edges[keep]

    return edges, lists_act, H_act


# =============================================================================
# Persistence sweep
# =============================================================================

def _persistence_sweep(X: np.ndarray, k_grid: np.ndarray,
                       p: MBCParams, k_i_template: np.ndarray, k_star: int,
                       active_mask: np.ndarray,
                       cores_full: Optional[np.ndarray] = None
                       ) -> List[Dict[str, Any]]:
    """For each k, build active-subgraph, return CC labels in full index space.

    Uses a copy of the params with the isolated-node fallback overridden
    by ``persistence_use_fallback`` so that the persistence sweep is a
    proper filtration over the mutual-kNN graphs, independent of the
    label graph at the canonical scale (which retains the fallback).
    """
    import dataclasses
    p_persist = dataclasses.replace(
        p,
        mutual_isolated_fallback=p.persistence_use_fallback,
        use_min_h_gate=False,
    )

    n = X.shape[0]
    active_idx = np.where(active_mask)[0]
    out: List[Dict[str, Any]] = []
    seen: Dict[Tuple[int, ...], Dict[str, Any]] = {}

    kmax_global = int(min(n - 1, max(int(k_i_template.max()) * 4, 4)))
    kmin_global = 2

    for k_target in k_grid:
        scale = float(k_target) / max(1.0, float(k_star))
        k_use = np.floor(scale * k_i_template).astype(int)
        k_use = np.clip(k_use, kmin_global, kmax_global)

        key = tuple(k_use[active_idx].tolist())
        if key in seen:
            prev = seen[key]
            out.append({
                "k": int(k_target),
                "K": prev["K"],
                "labels": prev["labels"],
                "n_edges": prev["n_edges"],
                "rescue_added": 0,
            })
            continue

        edges_act, _, _ = _build_graph(
            X, k_use, p_persist, active_idx, cores_full=cores_full)
        labs_act = _connected_components_from_edges(active_idx.size, edges_act)

        labs_full = np.full(n, -1, dtype=int)
        labs_full[active_idx] = labs_act

        K = int(np.unique(labs_act).size)
        entry = {
            "k": int(k_target),
            "K": K,
            "labels": labs_full,
            "n_edges": int(len(edges_act)),
            "rescue_added": 0,
        }
        seen[key] = entry
        out.append(entry)
    return out


# =============================================================================
# Persistence-based K and bracket
# =============================================================================

def _track_components(sweep: List[Dict[str, Any]],
                      active_idx: np.ndarray
                      ) -> Dict[int, Tuple[int, int, int]]:
    """For each canonical-id component ever seen, track (k_birth, k_death, size_at_birth).

    canonical id of a component at sweep t = min active-index in that component.
    A component 'dies' the first sweep step it is no longer a canonical id
    (i.e., its canonical point has been merged into a component with a smaller
    canonical id).
    """
    if not sweep:
        return {}

    canon_to_size_per_t: List[Dict[int, int]] = []
    for s in sweep:
        labs = s["labels"]
        # for each label, find min active idx and count (active idx only)
        labels_active = labs[active_idx]
        # active-local label -> min active-global index
        c2s: Dict[int, int] = {}
        c2c: Dict[int, int] = {}  # label -> canonical
        for ai_local, ai_global in enumerate(active_idx):
            lab = int(labels_active[ai_local])
            if lab < 0:
                continue
            if lab not in c2c or ai_global < c2c[lab]:
                c2c[lab] = int(ai_global)
            c2s[lab] = c2s.get(lab, 0) + 1
        canon_sizes: Dict[int, int] = {}
        for lab, canon in c2c.items():
            canon_sizes[canon] = c2s[lab]
        canon_to_size_per_t.append(canon_sizes)

    all_canons = set()
    for c2s in canon_to_size_per_t:
        all_canons.update(c2s.keys())

    out: Dict[int, Tuple[int, int, int]] = {}
    for c in all_canons:
        births = [t for t, c2s in enumerate(canon_to_size_per_t) if c in c2s]
        if not births:
            continue
        t_birth = births[0]
        sz_birth = canon_to_size_per_t[t_birth][c]
        deaths = [t for t in range(t_birth + 1, len(canon_to_size_per_t))
                  if c not in canon_to_size_per_t[t]]
        if deaths:
            k_death = sweep[deaths[0]]["k"]
        else:
            k_death = sweep[-1]["k"] + 1
        k_birth = sweep[t_birth]["k"]
        out[c] = (int(k_birth), int(k_death), int(sz_birth))
    return out


def _mass_K_at_step(labels_active: np.ndarray, gamma: float) -> int:
    """K_mass_gamma(k): smallest m such that the m largest active
    components cover at least ``gamma`` of the total active mass.

    Components with negative labels (pruned) are excluded from the mass.
    Returns 1 when the active set is empty or has a single component.
    """
    valid = labels_active[labels_active >= 0]
    if valid.size == 0:
        return 1
    _, counts = np.unique(valid, return_counts=True)
    if counts.size <= 1:
        return 1
    sizes = np.sort(counts)[::-1]
    target = float(gamma) * float(sizes.sum())
    cum = np.cumsum(sizes)
    return int(np.searchsorted(cum, target, side="left") + 1)


def _mass_bracket_from_zone(zone_mass_K: List[int], min_run: int
                            ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """From the per-step mass-K values across the zone, return
    ``(mass_bracket, mass_bracket_persistent)``.

    ``mass_bracket`` = (min, max) of K_mass(k) across the zone.
    ``mass_bracket_persistent`` = (min, max) of K_mass values that
    persist for at least ``min_run`` consecutive zone steps, dropping
    single-step flickers. Falls back to ``mass_bracket`` if no value
    persists long enough.
    """
    if not zone_mass_K:
        return (1, 1), (1, 1)
    lo, hi = min(zone_mass_K), max(zone_mass_K)

    persistent: List[int] = []
    run_val = zone_mass_K[0]
    run_len = 1
    for v in zone_mass_K[1:]:
        if v == run_val:
            run_len += 1
        else:
            if run_len >= min_run:
                persistent.append(run_val)
            run_val = v
            run_len = 1
    if run_len >= min_run:
        persistent.append(run_val)

    if persistent:
        plo, phi = min(persistent), max(persistent)
    else:
        plo, phi = lo, hi
    return (lo, hi), (plo, phi)


def _persistence_K_estimate(sweep: List[Dict[str, Any]],
                            n_active: int,
                            active_idx: np.ndarray,
                            k_low: int, k_high: int,
                            k_star: int,
                            min_size_frac: float = 0.005,
                            mass_gamma: float = 0.95,
                            mass_min_run: int = 2,
                            ) -> Tuple[int, int, int, int,
                                       Dict[int, np.ndarray], Dict[int, int],
                                       Tuple[int, int], bool, bool,
                                       Tuple[int, int], Tuple[int, int]]:
    """Bracket + K-selection (paper-canonical strict + practical fall-through).

    K_big(k) = number of components at scale k with size >= min_size, where
    min_size = max(ceil(min_size_frac * n_active), k_star). The 0.5% fraction
    is PAPER/main.tex's canonical value; the absolute floor of k_star is
    motivated by Theorem 1's cap-occupancy argument, which requires
    components to have at least Omega(k_star) = Omega(log n) samples for
    the radius-concentration bound to apply uniformly. Components below
    this floor are not certified by the theorem and are treated as dust.

    Bracket [K_low, K_high] = [min K_big(k), max K_big(k)] over the zone
    (this is exactly the paper's bracket definition).

    Two K's are computed and BOTH are returned:

    1. **K_persistent_strict** (paper Eq. 6): #{components alive throughout
       [k_low, k_high]} = #{c : k_birth(c) <= k_low AND k_death(c) > k_high}.
       Components are tracked by their canonical (min-index) representative.
       For abrupt-merge data this collapses to 1 (only the giant survives),
       which is the conservative answer the paper formalizes.

    2. **K_practical** (MBC1 Option C): if K_persistent_strict >= 2, use it;
       otherwise fall through to the mode of K_big(k) over the zone,
       restricted to K > 1. Among ties, prefer the larger K (under-merging
       is recoverable, over-merging is not).

    Returns (K_practical, K_persistent_strict, K_low_b, K_high_b,
             labels_per_K, k_for_K, raw_bracket, raw_monotone, big_monotone).

    raw_bracket = [min K_raw(k), max K_raw(k)] over the zone, where
    K_raw(k) is the unfiltered connected-component count. raw_monotone
    is True iff K_raw(k) is non-increasing in k across the zone (the
    filtration property the theorem assumes).
    """
    if not sweep:
        return 1, 1, 1, 1, {}, {}, (1, 1), True, True, (1, 1), (1, 1)

    # Floor at k_star (theorem's cap-occupancy requirement) OR 5 (small-n
    # safeguard) OR the fractional min_size_frac * |A| — strictest binds.
    # Setting min_size_frac=0 disables the percentage term and uses only
    # the theorem-mandated k_star floor.
    min_size = max(int(math.ceil(min_size_frac * n_active)), k_star, 5)

    in_zone = [s for s in sweep if k_low <= s["k"] <= k_high]
    if not in_zone:
        in_zone = sweep[:]

    def K_big(s) -> int:
        labs_act = s["labels"][active_idx]
        _, counts = np.unique(labs_act[labs_act >= 0], return_counts=True)
        return int((counts >= min_size).sum())

    K_at_zone = [(s["k"], K_big(s)) for s in in_zone]
    K_low_b = min(K for _, K in K_at_zone) if K_at_zone else 1
    K_high_b = max(K for _, K in K_at_zone) if K_at_zone else 1
    if K_low_b < 1: K_low_b = 1
    if K_high_b < K_low_b: K_high_b = K_low_b

    # Raw (unfiltered) bracket: range of K_raw(k) = #Comp(G_k^mut) over zone.
    # This is the actual 0-th persistence summary on the active set;
    # K_big_bracket above is the size-filtered practical version.
    raw_K_in_zone = [int(s["K"]) for s in in_zone]
    raw_low = min(raw_K_in_zone) if raw_K_in_zone else 1
    raw_high = max(raw_K_in_zone) if raw_K_in_zone else 1
    raw_bracket = (raw_low, raw_high)

    # Monotonicity check: K(k) should be non-increasing as k grows.
    # (Mutual-kNN edges only appear with larger k, so components only merge.)
    raw_monotone = all(raw_K_in_zone[i] >= raw_K_in_zone[i + 1]
                       for i in range(len(raw_K_in_zone) - 1))
    big_curve = [K for _, K in K_at_zone]
    big_monotone = all(big_curve[i] >= big_curve[i + 1]
                       for i in range(len(big_curve) - 1))

    # Mass-coverage companion: K_mass_gamma(k) at each zone step, and the
    # (min, max) range over the zone. mass_persistent additionally drops
    # K_mass values that don't persist for at least ``mass_min_run`` steps.
    mass_K_curve = [_mass_K_at_step(s["labels"][active_idx], mass_gamma)
                    for s in in_zone]
    mass_bracket, mass_bracket_persistent = _mass_bracket_from_zone(
        mass_K_curve, mass_min_run
    )

    # ---- K_persistent_strict (PAPER Eq. 6: alive throughout zone) ----
    # Track components by canonical (min-index) representative across sweep.
    # A component is "alive throughout the zone" iff its canonical id is born
    # at or before k_low AND dies strictly after k_high. Apply the same
    # min_size filter so dust components are not counted.
    lifetimes = _track_components(sweep, active_idx)
    K_persistent_strict = 0
    for c, (k_birth, k_death, sz_birth) in lifetimes.items():
        if sz_birth >= min_size and k_birth <= k_low and k_death > k_high:
            K_persistent_strict += 1
    if K_persistent_strict < 1:
        K_persistent_strict = 1  # at least the trivial single component

    # ---- K_practical (MBC1 Option C) ----
    # If the strict rule returned >= 2, the paper's K_persistent is itself
    # informative — keep it. Otherwise (typical for non-separable data
    # where only the giant is alive throughout) fall through to the mode of
    # K_big(k) over the zone, restricted to K > 1.
    if K_persistent_strict >= 2:
        K_practical = K_persistent_strict
    else:
        counter: Dict[int, int] = {}
        for _, K in K_at_zone:
            counter[K] = counter.get(K, 0) + 1
        non_trivial = {K: c for K, c in counter.items() if K > 1}
        if non_trivial:
            max_count = max(non_trivial.values())
            best_Ks = [K for K, c in non_trivial.items() if c == max_count]
            # Prefer larger K on ties (under-merging recoverable).
            K_practical = int(max(best_Ks))
        else:
            K_practical = 1

    K_hat = K_practical

    # Build labels-per-K. For each K in K_candidates, find a sweep step that
    # naturally exhibits K_big = K, then canonicalize to exactly K labels.
    def _canonicalize_to_K(labels_full: np.ndarray, K_target: int) -> Optional[np.ndarray]:
        out = labels_full.copy()
        labs_act = out[active_idx]
        unique = np.unique(labs_act[labs_act >= 0])
        n_unique = len(unique)

        if K_target == 1:
            out_new = np.full_like(out, -1)
            out_new[active_idx[labs_act >= 0]] = 0
            return out_new

        if n_unique < K_target:
            return None

        if n_unique == K_target:
            remap = {int(old): new for new, old in enumerate(unique)}
            for ai_global in active_idx:
                v = int(out[ai_global])
                if v >= 0:
                    out[ai_global] = remap[v]
            return out

        # n_unique > K_target: keep top-K_target by size, demote rest
        counts = np.array([int((labs_act == u).sum()) for u in unique])
        order = np.argsort(-counts)
        keep = set(int(unique[order[i]]) for i in range(K_target))
        for ai_local, ai_global in enumerate(active_idx):
            lab = int(labs_act[ai_local])
            if lab >= 0 and lab not in keep:
                out[ai_global] = -1
        kept_sorted = sorted(keep)
        remap = {old: new for new, old in enumerate(kept_sorted)}
        for ai_global in active_idx:
            v = int(out[ai_global])
            if v >= 0:
                out[ai_global] = remap[v]
        return out

    labels_per_K: Dict[int, np.ndarray] = {}
    k_for_K: Dict[int, int] = {}

    K_candidates = set()
    for s in sweep:
        K_candidates.add(int(s["K"]))
        K_candidates.add(int(K_big(s)))
    K_candidates.add(int(K_hat))
    K_candidates.add(int(K_low_b))
    K_candidates.add(int(K_high_b))
    K_candidates.discard(0)
    K_candidates.add(1)

    median_k = int(np.median([s["k"] for s in sweep])) if sweep else 0
    for K in K_candidates:
        if K < 1:
            continue
        # Prefer steps where K_big is exactly K (the natural K-step).
        # Otherwise, fall back to any step with raw K >= K_target.
        cands_natural = [s for s in sweep if K_big(s) == K]
        if cands_natural:
            cands_natural.sort(key=lambda s: abs(int(s["k"]) - median_k))
            best_step = cands_natural[0]
        else:
            cands_raw = [s for s in sweep if int(s["K"]) >= K]
            if not cands_raw:
                cands_raw = [max(sweep, key=lambda s: int(s["K"]))]
            cands_raw.sort(key=lambda s: (int(s["K"]) - K, abs(int(s["k"]) - median_k)))
            best_step = cands_raw[0]
        canon = _canonicalize_to_K(best_step["labels"], K)
        if canon is not None:
            labels_per_K[int(K)] = canon
            k_for_K[int(K)] = int(best_step["k"])

    return (int(K_hat), int(K_persistent_strict),
            int(K_low_b), int(K_high_b), labels_per_K, k_for_K,
            raw_bracket, bool(raw_monotone), bool(big_monotone),
            (int(mass_bracket[0]), int(mass_bracket[1])),
            (int(mass_bracket_persistent[0]), int(mass_bracket_persistent[1])))


# =============================================================================
# rho_hat (diagnostic only)
# =============================================================================

def _estimate_rho_hat(X: np.ndarray, edges: np.ndarray, H: np.ndarray,
                      labels_active: np.ndarray, active_idx: np.ndarray,
                      h_quantile: float = 0.5
                      ) -> Tuple[float, float, str, Dict[str, float]]:
    """Estimate rho = Delta / h_max from a coarse partition at k*.

    Returns (rho_hat, offset_proxy, source, panel) where panel reports the
    estimate at multiple radius quantiles for sensitivity analysis.
    The headline rho_hat uses ``h_quantile`` (default 0.5 = median).

    Sources used to estimate Delta (the numerator):
      1. min cross-component sample distance from the coarse mutual-kNN
         partition at k*, when the graph has >= 2 components;
      2. largest edge in the minimum spanning tree of the active set,
         when the coarse graph is connected (single component);
      3. fallback constant rho = 1 with source 'fallback'.
    """
    H_pos = H[H > 0]
    if H_pos.size == 0:
        return 1.0, 1.0, "fallback", {}
    panel = {
        "h_q50": float(np.quantile(H_pos, 0.5)),
        "h_q75": float(np.quantile(H_pos, 0.75)),
        "h_q90": float(np.quantile(H_pos, 0.90)),
        "h_q95": float(np.quantile(H_pos, 0.95)),
    }
    h_proxy = float(np.quantile(H_pos, h_quantile))
    h_proxy = max(h_proxy, 1e-12)

    n_comp = int(np.unique(labels_active).size)

    if n_comp >= 2:
        from scipy.spatial.distance import cdist
        rng = np.random.default_rng(0)
        unique_labs = np.unique(labels_active)
        sampled = {}
        for L in unique_labs:
            idx_local = np.where(labels_active == L)[0]
            if len(idx_local) > 100:
                idx_local = rng.choice(idx_local, 100, replace=False)
            sampled[int(L)] = idx_local
        L_list = list(sampled.keys())
        min_cross = np.inf
        for ii in range(len(L_list)):
            for jj in range(ii + 1, len(L_list)):
                A = X[active_idx[sampled[L_list[ii]]]]
                B = X[active_idx[sampled[L_list[jj]]]]
                m = float(cdist(A, B).min())
                if m < min_cross:
                    min_cross = m
        if np.isfinite(min_cross) and min_cross > 0:
            return (float(min_cross / h_proxy), float(min_cross),
                    "min_cross_component", panel)

    # When the coarse mutual-kNN graph at k* is connected, no
    # cross-component pair exists to estimate Delta. We return rho_hat
    # = -1 to signal "non-separable" rather than guessing from a single
    # MST edge: the largest MST edge is not a principled estimator of
    # the inter-component bottleneck on data without a clear cut, and
    # using it pushes contaminated or high-D embeddings into a falsely
    # confident "separable" regime.
    if edges.size > 0:
        d_in = np.linalg.norm(
            X[active_idx[edges[:, 0]]] - X[active_idx[edges[:, 1]]], axis=1)
        return -1.0, float(np.max(d_in)), "single_component", panel

    return 1.0, h_proxy, "fallback", panel


def _compute_bracket_endpoints(rho_hat: float, A_anchor: float, p: MBCParams,
                               d: int) -> Tuple[float, float, str]:
    if not p.use_data_driven_bracket:
        return (A_anchor / p.A_factor_fallback,
                A_anchor * p.A_factor_fallback, "fixed")

    C_under_anchor, C_over_anchor = _threshold_constants(
        A_anchor, d, p.epsilon_no_bridge, p.a_collar
    )
    c_over = C_over_anchor / max(A_anchor ** (1.0 / d), 1e-9)
    c_under = C_under_anchor / max(A_anchor ** (1.0 / d), 1e-9)

    if rho_hat < 0:
        regime = "non_separable"
        A_low = A_anchor * p.bracket_A_min_ratio
        A_high = A_anchor * 1.10
    else:
        rho_inv = max(rho_hat, 1e-6)
        A_star = (rho_inv / max(c_over, 1e-9)) ** d
        A_dstar = (rho_inv / max(c_under, 1e-9)) ** d

        if rho_hat >= C_over_anchor:
            regime = "separable"
            A_low = A_anchor * 0.85
            A_high = A_anchor * 1.15
        elif rho_hat <= C_under_anchor:
            regime = "non_separable"
            A_low = A_anchor * p.bracket_A_min_ratio
            A_high = A_anchor * 1.10
        else:
            regime = "transitional"
            A_low = A_star
            A_high = A_dstar

    A_low = max(A_low, A_anchor * p.bracket_A_min_ratio)
    A_high = min(A_high, A_anchor * p.bracket_A_max_ratio)
    A_low = min(A_low, A_anchor)
    A_high = max(A_high, A_anchor)
    if A_high <= A_low:
        A_low, A_high = A_anchor * 0.9, A_anchor * 1.1
    return float(A_low), float(A_high), regime


# =============================================================================
# Main entry point
# =============================================================================

def mbc_cluster(X: np.ndarray, p: Optional[MBCParams] = None,
                seed: int = 0,
                *,
                standardize: bool = False,
                pca_mode: str = "none") -> MBCResult:
    """Cluster X with the MBC bracket estimator.

    Parameters
    ----------
    X : (n, D) array
        Input feature matrix.
    p : MBCParams, optional
        Algorithm parameters; defaults via :class:`MBCParams`.
    seed : int
        Random seed (used only by the rho-hat per-component subsampler).
    standardize : bool, default False
        If True, z-score each feature before clustering. The canonical
        method does not require standardization, so it is opt-in.
    pca_mode : {"none", "project_90", "project_fixed"}, default "none"
        Optional PCA projection step applied AFTER standardization (if
        any). ``"project_90"`` projects onto the smallest number of
        components explaining 90% of variance (capped at 64);
        ``"project_fixed"`` projects onto a fixed number set via
        ``p.pca_n_components`` (currently uses the d_eff estimator).
        ``"none"`` keeps the input space.
    """
    if p is None:
        p = MBCParams()
    X = np.asarray(X, dtype=float)
    n, D = X.shape

    metric_space = "input"
    if standardize:
        X = StandardScaler(with_std=True).fit_transform(X)
        metric_space = "standardized"
    if pca_mode == "project_90":
        d_eff_est = _effective_dimensionality(X)
        if d_eff_est < D:
            n_comp = min(d_eff_est, max(1, n - 1), D)
            pca = PCA(n_components=n_comp, svd_solver="randomized",
                      random_state=0)
            X = pca.fit_transform(X)
            metric_space = "pca_projected"
            n, D = X.shape
    elif pca_mode == "project_fixed":
        d_eff_est = _effective_dimensionality(X)
        n_comp = min(d_eff_est, max(1, n - 1), D)
        pca = PCA(n_components=n_comp, svd_solver="randomized",
                  random_state=0)
        X = pca.fit_transform(X)
        metric_space = "pca_projected_fixed"
        n, D = X.shape
    elif pca_mode != "none":
        raise ValueError(f"Unknown pca_mode: {pca_mode!r}")

    # ---- Pilot scale and local-k schedule ----
    d_eff = _effective_dimensionality(X, max_comp=int(p.d_eff_cap))
    k_star = _choose_k_star(n, p.delta, p.A_coef)
    log_term = math.log(max(3.0, 4.0 * n) / max(1e-12, p.delta))
    kmin = max(2, int(math.ceil(p.kmin_log_multiplier * log_term)))
    kmax = min(n - 1, int(math.floor(p.kmax_factor * float(k_star))))

    d_pilot, ind_pilot = _fit_nn(X, k_star)
    H_pilot = d_pilot[:, min(k_star, d_pilot.shape[1] - 1)]

    # ---- Density pruning ----
    active_mask, pruned_mask, tau_noise = _density_prune(H_pilot, p)
    n_active = int(active_mask.sum())
    active_idx = np.where(active_mask)[0]

    if p.use_local_k:
        k_i = _local_k_schedule(H_pilot, d_eff, k_star, kmin, kmax,
                                p.enforce_k_floor)
    else:
        k_i = np.full(n, int(k_star), dtype=int)
        k_i = np.clip(k_i, kmin, kmax)

    # ---- Wider full-data kNN for the noise-reassign step ----
    K_for_fit = int(min(n - 1, max(int(k_i.max()),
                                    int(math.ceil(p.bracket_A_max_ratio * k_star)),
                                    kmax)))
    d_full, ind_full = _fit_nn(X, K_for_fit)

    # ---- Main graph at k_i on active subset ----
    edges_main, lists_main, H_main = _build_graph(
        X, k_i, p, active_idx, cores_full=H_pilot)
    labels_active_main = _connected_components_from_edges(n_active, edges_main)
    K_main_active = int(np.unique(labels_active_main).size)

    # ---- rho_hat estimation ----
    rho_hat, offset_proxy, rho_source, rho_panel = _estimate_rho_hat(
        X, edges_main, H_main, labels_active_main, active_idx,
        h_quantile=p.rho_h_quantile,
    )
    C_under, C_over = _threshold_constants(
        p.A_coef, d_eff, p.epsilon_no_bridge, p.a_collar
    )
    A_low, A_high, regime = _compute_bracket_endpoints(
        rho_hat, p.A_coef, p, d_eff
    )

    k_low = max(2, int(math.ceil(A_low * log_term)))
    k_high = min(n - 1, int(math.ceil(A_high * log_term)))
    if k_high < k_low:
        k_low, k_high = k_high, k_low
    k_low = max(k_low, kmin)
    k_high = max(k_high, k_low + 1)

    # ---- Sweep ----
    n_sweep = min(p.bracket_n_sweep, max(2, k_high - k_low + 1))
    k_grid = np.unique(np.linspace(k_low, k_high, n_sweep).astype(int))
    bracket_sweep = _persistence_sweep(
        X, k_grid, p, k_i, k_star, active_mask, cores_full=H_pilot
    )

    # ---- K-selection: paper-strict + practical fall-through ----
    (K_persistent, K_persistent_strict, K_low_b, K_high_b,
     labels_per_K, k_for_K,
     raw_bracket, raw_monotone, big_monotone,
     mass_bracket, mass_bracket_persistent) = _persistence_K_estimate(
        bracket_sweep, n_active, active_idx, k_low, k_high,
        k_star=k_star,
        min_size_frac=p.cluster_min_size_frac,
        mass_gamma=p.mass_gamma,
        mass_min_run=p.mass_min_run,
    )

    # Pick the labels corresponding to K_persistent (or closest available)
    if K_persistent in labels_per_K:
        labels_full_main = labels_per_K[K_persistent]
    else:
        labs_at_kstar = None
        for s in bracket_sweep:
            if s["k"] >= k_star:
                labs_at_kstar = s["labels"]
                break
        if labs_at_kstar is None and bracket_sweep:
            labs_at_kstar = bracket_sweep[-1]["labels"]
        if labs_at_kstar is None:
            labs_at_kstar = np.full(n, -1, dtype=int)
            labs_at_kstar[active_idx] = labels_active_main
        labels_full_main = labs_at_kstar

    # ---- Reassign pruned points (or leave as -1) ----
    if pruned_mask.sum() > 0 and p.noise_reassign:
        labels_with_noise = _reassign_pruned(
            X, labels_full_main, active_mask, pruned_mask, H_pilot, p, ind_full
        )
    else:
        labels_with_noise = labels_full_main.copy()

    # Build labels for each K in two flavors:
    #   bracket_labels_with_noise:  pruned points that the local-radius
    #     reassignment couldn't place are left as -1 (DBSCAN-style noise).
    #   bracket_labels:             every point is force-assigned to its
    #     nearest cluster, so np.unique(...).size == K.
    bracket_labels_full: Dict[int, np.ndarray] = {}
    bracket_labels_noise: Dict[int, np.ndarray] = {}
    for K, labs in labels_per_K.items():
        if pruned_mask.sum() > 0 and p.noise_reassign:
            with_noise = _reassign_pruned(X, labs, active_mask, pruned_mask,
                                          H_pilot, p, ind_full)
        else:
            with_noise = labs.copy()
        bracket_labels_noise[int(K)] = with_noise
        if p.bracket_labels_force_assign:
            bracket_labels_full[int(K)] = _force_assign_all(
                X, with_noise, ind_full, int(K)
            )
        else:
            bracket_labels_full[int(K)] = with_noise.copy()

    if K_persistent not in bracket_labels_full:
        bracket_labels_noise[K_persistent] = labels_with_noise.copy()
        if p.bracket_labels_force_assign:
            bracket_labels_full[K_persistent] = _force_assign_all(
                X, labels_with_noise, ind_full, int(K_persistent)
            )
        else:
            bracket_labels_full[K_persistent] = labels_with_noise.copy()
        k_for_K[K_persistent] = k_for_K.get(K_persistent, k_star)

    # ---- Bookkeeping ----
    bracket_curve = [(s["k"], s["K"]) for s in bracket_sweep]

    n_clusters = int(np.unique(labels_with_noise[labels_with_noise >= 0]).size)
    noise_count = int((labels_with_noise == -1).sum())

    H_pos = H_main[H_main > 0]
    H_med = float(np.median(H_pos)) if H_pos.size else 0.0
    H_min_v = float(H_pos.min()) if H_pos.size else 0.0
    H_max_v = float(H_main.max()) if H_main.size else 0.0

    return MBCResult(
        labels=labels_with_noise,
        n_clusters=n_clusters,
        bracket=(K_low_b, K_high_b),
        bracket_K_curve=bracket_curve,
        bracket_labels=bracket_labels_full,
        bracket_k_for_K={int(K): int(v) for K, v in k_for_K.items()},
        bracket_sweep=bracket_sweep,
        k_star=int(k_star),
        k_low=int(k_low),
        k_high=int(k_high),
        rho_hat=float(rho_hat),
        rho_hat_lower=float(offset_proxy),
        C_lower=float(C_under),
        C_upper=float(C_over),
        A_low=float(A_low),
        A_high=float(A_high),
        d_eff=int(d_eff),
        n_edges_final=int(len(edges_main)),
        n_edges_remove_only=int(len(edges_main)),
        dtm_rescue_used=False,
        n_rescued_edges=0,
        noise_count=noise_count,
        H_median=H_med,
        H_min=H_min_v,
        H_max=H_max_v,
        info=dict(
            algo=("MBC: mutual-kNN + density-prune + persistence-K"
                  if p.candidate_mode == "mutual"
                  else "MBC: union-kNN + density-prune + persistence-K"),
            n=int(n), D=int(D),
            kmin=int(kmin), kmax=int(kmax),
            n_candidates=int(len(edges_main)),
            n_edges_eucl=int(len(edges_main)),
            n_active=int(n_active),
            n_pruned=int(pruned_mask.sum()),
            tau_noise=float(tau_noise),
            seed=int(seed),
            rho_source=rho_source,
            rho_h_quantile=float(p.rho_h_quantile),
            rho_panel=rho_panel,
            regime=regime,
            K_persistent=int(K_persistent),
            K_persistent_strict=int(K_persistent_strict),
            candidate_mode=p.candidate_mode,
            density_pruning=bool(p.use_density_pruning),
            # ---- bracket diagnostics ----
            bracket_raw=(int(raw_bracket[0]), int(raw_bracket[1])),
            bracket_big=(int(K_low_b), int(K_high_b)),
            bracket_raw_vs_big_gap=int(raw_bracket[1] - K_high_b),
            raw_curve_monotone=bool(raw_monotone),
            big_curve_monotone=bool(big_monotone),
            # mass-coverage companion (gamma, run-length stable variant)
            bracket_mass=(int(mass_bracket[0]), int(mass_bracket[1])),
            bracket_mass_persistent=(int(mass_bracket_persistent[0]),
                                     int(mass_bracket_persistent[1])),
            mass_gamma=float(p.mass_gamma),
            mass_min_run=int(p.mass_min_run),
            pruning_fraction=float(pruned_mask.sum()) / max(1, n),
            persistence_use_fallback=bool(p.persistence_use_fallback),
            use_min_h_gate=bool(p.use_min_h_gate),
            metric_space=metric_space,
        ),
        bracket_labels_with_noise=bracket_labels_noise,
    )


def mbc_cluster_legacy(X: np.ndarray, p: Optional[MBCParams] = None,
                       seed: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
    res = mbc_cluster(X, p, seed=seed)
    info = dict(res.info)
    info.update(dict(
        algo=res.info.get("algo"),
        k_star=res.k_star,
        k_low=res.k_low,
        k_high=res.k_high,
        d_eff=res.d_eff,
        dtm_rescue_used=res.dtm_rescue_used,
        K_CI=tuple(res.bracket),
        edges=res.n_edges_final,
        K=res.n_clusters,
        noise_count=res.noise_count,
        rho_hat=res.rho_hat,
        regime=res.info.get("regime"),
        sweep_curve=res.bracket_K_curve,
    ))
    return res.labels, info


def standardize_then_mbc(X: np.ndarray, p: Optional[MBCParams] = None,
                          seed: int = 0) -> MBCResult:
    """Backward-compatible alias: standardize + PCA project to d_eff (90% EVR).

    Equivalent to ``mbc_cluster(X, p, seed=seed,
    standardize=True, pca_mode='project_90')``.
    """
    return mbc_cluster(X, p, seed=seed,
                       standardize=True, pca_mode="project_90")


def get_labels_for_K(res: "MBCResult", K_target: int) -> Optional[np.ndarray]:
    """Return labels for exactly K_target clusters.

    Behavior matches what's stored in `bracket_labels`:
      - If `bracket_labels[K_target]` exists, return that array directly
        (already force-assigned: every point in 0..K-1, no -1).
      - Otherwise canonicalize down from the smallest stored K >= K_target
        by merging the smallest excess clusters into the largest, so the
        returned array has exactly K_target unique values (still no -1).
      - If K_target > max(stored) or K_target < 1, return None.

    Use `res.bracket_labels_with_noise[K]` to inspect the noise-aware
    version of stored Ks.
    """
    if K_target in res.bracket_labels:
        return res.bracket_labels[K_target]
    Ks_avail = sorted(res.bracket_labels.keys())
    if not Ks_avail:
        return None
    if K_target < 1:
        return None
    if K_target > max(Ks_avail):
        return None

    above = [K for K in Ks_avail if K >= K_target]
    if not above:
        return None
    src_K = min(above)
    src = res.bracket_labels[src_K].copy()

    if K_target == 1:
        return np.zeros_like(src)

    # Merge the smallest (src_K - K_target) clusters into the largest one.
    # This preserves the no-(-1) invariant.
    n_to_merge = src_K - K_target
    unique, counts = np.unique(src, return_counts=True)
    order = np.argsort(counts)  # ascending
    smallest = [int(unique[order[i]]) for i in range(n_to_merge)]
    largest = int(unique[np.argmax(counts)])
    for s in smallest:
        src[src == s] = largest
    # Relabel to contiguous 0..K_target-1
    new_unique = np.unique(src)
    remap = {int(old): new for new, old in enumerate(new_unique)}
    return np.array([remap[int(v)] for v in src], dtype=src.dtype)


def summarize(res: MBCResult) -> str:
    rho_str = ("n/a (single component)" if res.rho_hat < 0
               else f"{res.rho_hat:.3f}")
    bracket_K_list = sorted(res.bracket_labels.keys())
    bracket_label_summary = ", ".join(
        f"K={K}@k={res.bracket_k_for_K.get(K, '?')}" for K in bracket_K_list
    )
    lines = [
        f"=== MBC Result ===",
        f"  n={res.info.get('n')}, D={res.info.get('D')}, d_eff={res.d_eff}",
        f"  algo: {res.info.get('algo')}",
        f"  active/pruned: {res.info.get('n_active')}/{res.info.get('n_pruned')} "
        f"(tau_noise={res.info.get('tau_noise', 0):.3f})",
        f"  k* = {res.k_star}     (kmin={res.info.get('kmin')}, "
        f"kmax={res.info.get('kmax')})",
        f"  bracket k:  [{res.k_low}, {res.k_high}]",
        f"  rho_hat = {rho_str}    "
        f"[theory C_low={res.C_lower:.3f}, C_high={res.C_upper:.3f}]",
        f"  regime: {res.info.get('regime')}    "
        f"(rho source: {res.info.get('rho_source')})",
        f"  A_low / A_high = {res.A_low:.3f} / {res.A_high:.3f}",
        f"  H median/min/max = {res.H_median:.4f} / {res.H_min:.4f} / "
        f"{res.H_max:.4f}",
        f"  edges (final): {res.n_edges_final}",
        f"  noise count (post-reassign): {res.noise_count}",
        f"  K (excluding noise): {res.n_clusters}",
        f"  K_practical (n_clusters): {res.info.get('K_persistent')}    "
        f"K_persistent_strict (paper Eq. 6): "
        f"{res.info.get('K_persistent_strict')}",
        f"  Bracket on K: [{res.bracket[0]}, {res.bracket[1]}]",
        f"  Sweep curve: {res.bracket_K_curve}",
        f"  Labels stored for: {bracket_label_summary}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_per = 200
    X1 = rng.normal(loc=[0, 0], scale=0.3, size=(n_per, 2))
    X2 = rng.normal(loc=[5, 5], scale=0.3, size=(n_per, 2))
    X3 = rng.normal(loc=[0, 5], scale=0.3, size=(n_per, 2))
    X = np.vstack([X1, X2, X3])
    res = standardize_then_mbc(X)
    print(summarize(res))