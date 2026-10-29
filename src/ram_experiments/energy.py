"""Graph energy, nuclear-norm surrogate and spectral diagnostics.

All public measurement functions accept NumPy arrays and accumulate in
``float64``. They never threshold the spectrum: effective rank and numerical
rank are distinct objects and are reported separately.
"""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Sequence

import numpy as np

Array = np.ndarray


def as_matrices(weights: Sequence[Array | object]) -> list[Array]:
    """Convert a sequence of array-like matrices to contiguous float64 arrays."""
    matrices = []
    for weight in weights:
        array = np.asarray(weight, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(f"expected a 2-D weight matrix, got shape {array.shape}")
        matrices.append(np.ascontiguousarray(array))
    return matrices


def nuclear_norm(matrix: Array) -> float:
    """Sum of singular values of a matrix."""
    array = np.asarray(matrix, dtype=np.float64)
    return float(np.linalg.svd(array, compute_uv=False, full_matrices=False).sum())


def omega(weights: Sequence[Array | object]) -> float:
    """Global nuclear-norm surrogate ``Omega(W) = 2 * sum_l ||W_l||_*``."""
    return float(2.0 * sum(nuclear_norm(w) for w in as_matrices(weights)))


def graph_adjacency(weights: Sequence[Array | object]) -> Array:
    """Signed symmetric adjacency of consecutive layers, without biases.

    For ``W_l`` of shape ``(n_l, n_{l-1})`` the block ``A[l, l-1] = W_l`` and
    ``A[l-1, l] = W_l.T``; all other blocks are zero. Entrywise absolute values
    are *not* taken.
    """
    matrices = as_matrices(weights)
    if not matrices:
        raise ValueError("at least one weight matrix is required")
    widths = [matrices[0].shape[1]]
    widths.extend(matrix.shape[0] for matrix in matrices)
    for index, matrix in enumerate(matrices, start=1):
        expected = (widths[index], widths[index - 1])
        if matrix.shape != expected:
            raise ValueError(f"layer {index} has shape {matrix.shape}, expected {expected}")

    total = sum(widths)
    adjacency = np.zeros((total, total), dtype=np.float64)
    offsets = np.cumsum([0, *widths])
    for index, matrix in enumerate(matrices, start=1):
        row = slice(offsets[index], offsets[index + 1])
        col = slice(offsets[index - 1], offsets[index])
        adjacency[row, col] = matrix
        adjacency[col, row] = matrix.T
    return adjacency


def graph_energy(weights: Sequence[Array | object]) -> float:
    """Nuclear norm of the signed adjacency matrix, ``E(A) = ||A||_*``."""
    return nuclear_norm(graph_adjacency(weights))


def energy_ratio(weights: Sequence[Array | object]) -> float | None:
    """Return ``E(A) / Omega`` or ``None`` when ``Omega`` is zero."""
    surrogate = omega(weights)
    if surrogate == 0.0:
        return None
    return graph_energy(weights) / surrogate


def spectral_dimension(weights: Sequence[Array | object]) -> int:
    """``D = sum_l min(n_l, n_{l-1})``."""
    return int(sum(min(matrix.shape) for matrix in as_matrices(weights)))


def spectral_summary(matrix: Array) -> dict[str, float | int | list[float]]:
    """Complete per-layer spectrum summary without thresholding."""
    values = np.linalg.svd(
        np.asarray(matrix, dtype=np.float64), compute_uv=False, full_matrices=False
    )
    total = float(values.sum())
    if total == 0.0:
        effective_rank = 0.0
    else:
        probabilities = values / total
        positive = probabilities[probabilities > 0]
        effective_rank = float(np.exp(-np.sum(positive * np.log(positive))))
    return {
        "shape": list(np.shape(matrix)),
        "n_singular_values": int(values.size),
        "sigma_max": float(values.max()) if values.size else 0.0,
        "sigma_min": float(values.min()) if values.size else 0.0,
        "nuclear_norm": total,
        "frobenius_norm": float(np.linalg.norm(np.asarray(matrix, dtype=np.float64))),
        "effective_rank": effective_rank,
        "spectrum": [float(value) for value in values],
    }


def omega_summary(weights: Sequence[Array | object]) -> dict[str, object]:
    """Per-layer and global spectral summary."""
    matrices = as_matrices(weights)
    layers = [spectral_summary(matrix) for matrix in matrices]
    return {
        "omega": omega(matrices),
        "D": spectral_dimension(matrices),
        "layers": layers,
    }


def violation(weights: Sequence[Array | object], tau: float) -> float:
    """Raw feasibility violation ``max(Omega(W) - tau, 0)``."""
    if tau < 0:
        raise ValueError("tau must be non-negative")
    return max(omega(weights) - tau, 0.0)


def is_feasible(
    weights: Sequence[Array | object], tau: float, tolerance: float | None = None
) -> bool:
    from ram_experiments.config import feasibility_tolerance

    tol = feasibility_tolerance(tau) if tolerance is None else tolerance
    return violation(weights, tau) <= tol


def weights_sha256(weights: Sequence[Array | object]) -> str:
    """Stable hash of the matrix values (shape and dtype included)."""
    digest = hashlib.sha256()
    for matrix in as_matrices(weights):
        digest.update(str(matrix.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(matrix, dtype="<f8").tobytes())
    return digest.hexdigest()


def perturbation_bound(
    left: Sequence[Array | object],
    right: Sequence[Array | object],
    rank: int | None = None,
    dimension: int | None = None,
) -> float:
    """``2 sqrt(R) ||W - V||_F`` (or the conservative ``2 sqrt(D)`` version)."""
    a = as_matrices(left)
    b = as_matrices(right)
    if len(a) != len(b):
        raise ValueError("both weight lists must have the same number of layers")
    difference = [x - y for x, y in zip(a, b, strict=True)]
    frobenius = float(np.sqrt(sum(np.sum(d**2) for d in difference)))
    if rank is not None:
        factor = float(np.sqrt(max(rank, 0)))
    elif dimension is not None:
        factor = float(np.sqrt(max(dimension, 0)))
    else:
        factor = float(np.sqrt(max(spectral_dimension(a), 0)))
    return float(2.0 * factor * frobenius)


def dimension_counts(widths: Sequence[int]) -> dict[str, int]:
    """Spectral dimension ``D`` and parameter counts for an MLP width list."""
    weights = 0
    parameters = 0
    spectral = 0
    for fan_in, fan_out in itertools.pairwise(widths):
        weights += fan_in * fan_out
        parameters += fan_in * fan_out + fan_out
        spectral += min(fan_in, fan_out)
    return {"weights": weights, "parameters": parameters, "D": spectral}
