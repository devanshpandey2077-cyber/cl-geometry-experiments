from __future__ import annotations
from itertools import combinations
import numpy as np
from scipy.sparse.linalg import svds

def subspace(x, rank, tol=1e-7):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Features must be a finite matrix")
    x = x - x.mean(axis=0, keepdims=True)
    n, d = x.shape
    if rank < 1 or rank >= min(n, d):
        raise ValueError(f"Rank {rank} needs more independent rows/features than {x.shape}")
    # Compute one extra singular value to record the cutoff gap.
    k = min(rank + 1, min(n, d) - 1)
    if min(n, d) <= 64 or k >= min(n, d) - 1:
        _, s, vt = np.linalg.svd(x, full_matrices=False)
    else:
        # Fixed solver seed: solver randomness must not masquerade as training drift.
        _, s, vt = svds(x, k=k, which="LM", tol=tol, random_state=0)
        order = np.argsort(s)[::-1]
        s, vt = s[order], vt[order]
    if s[rank - 1] <= max(s[0], 1.0) * 1e-10:
        raise ValueError("Requested rank exceeds the observed numerical rank")
    energy = float(np.square(x).sum())
    extra = float(s[rank]) if len(s) > rank else None
    info = {"singular_values": [float(v) for v in s[:rank + 1]],
            "explained_variance": float(np.square(s[:rank]).sum() / energy),
            "cutoff_gap": float(s[rank - 1] - extra) if extra is not None else None}
    return vt[:rank].T, info

def affinity(u, v):
    if u.shape != v.shape:
        raise ValueError("Compare equal-rank bases in the same feature coordinates")
    return float(np.clip(np.square(u.T @ v).sum() / u.shape[1], 0, 1))

def affinity_matrix(features, rank):
    tasks = sorted(features)
    bases, diagnostics = {}, {}
    for task in tasks:
        bases[task], diagnostics[task] = subspace(features[task], rank)
    a = np.eye(len(tasks), dtype=np.float64)
    for i, j in combinations(range(len(tasks)), 2):
        a[i, j] = a[j, i] = affinity(bases[tasks[i]], bases[tasks[j]])
    return tasks, a, diagnostics

def pair_vector(a):
    pairs = list(combinations(range(a.shape[0]), 2))
    return pairs, np.asarray([a[i, j] for i, j in pairs])

def differences(values):
    comparisons = list(combinations(range(len(values)), 2))
    return comparisons, np.asarray([values[i] - values[j] for i, j in comparisons])

def group_bootstrap_indices(groups, rng):
    groups = np.asarray(groups)
    unique = np.unique(groups)
    buckets = {g: np.flatnonzero(groups == g) for g in unique}
    chosen = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([buckets[g] for g in chosen])


def joint_group_bootstrap_indices(task_groups, rng):
    """One multiplicity per global source group, shared across tasks and checkpoints."""
    tasks = sorted(task_groups)
    unique = np.unique(np.concatenate([np.asarray(task_groups[t]) for t in tasks]))
    selected = rng.integers(len(unique), size=len(unique))
    counts = np.bincount(selected, minlength=len(unique))
    result = {}
    for task in tasks:
        groups = np.asarray(task_groups[task])
        row_counts = counts[np.searchsorted(unique, groups)]
        result[task] = np.repeat(np.arange(len(groups)), row_counts)
        if len(result[task]) == 0:
            raise ValueError("A group-bootstrap draw left a task empty; increase probe sample size")
    return result

def simultaneous_intervals(point, boots, alpha=0.05):
    """Approximate bootstrap max-standardized-error intervals over the entire supplied family."""
    point = np.asarray(point, float)
    boots = np.asarray(boots, float)
    if boots.shape[1:] != point.shape or len(boots) < 20:
        raise ValueError("Need >=20 aligned bootstrap draws and compatible shapes")
    se = boots.std(axis=0, ddof=1)
    if not np.isfinite(point).all() or not np.isfinite(boots).all():
        raise ValueError("Non-finite bootstrap estimates")
    constant_but_shifted = (se <= 1e-12) & (np.max(np.abs(boots-point),axis=0) > 1e-8)
    if constant_but_shifted.any():
        raise ValueError("Degenerate bootstrap has zero variance but differs from the point estimate")
    scale = np.where(se > 1e-12, se, 1.0)
    errors = np.abs((boots - point) / scale)
    q = float(np.quantile(errors.reshape(len(boots), -1).max(axis=1), 1 - alpha))
    return point - q * se, point + q * se, {"critical_value": q, "alpha": alpha,
          "family_size": int(point.size), "resamples": len(boots),
          "method": "approximate paired group bootstrap max-standardized-error"}

def inversion_flags(lower, upper, margin=0.01):
    if lower.shape[0] != 2:
        raise ValueError("Expected initial and final differences")
    return ((lower[0] > margin) & (upper[1] < -margin)) | (
            (upper[0] < -margin) & (lower[1] > margin))

