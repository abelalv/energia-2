"""Exact joint nuclear-norm projection and related spectral operations.

The projection is computed in ``float64`` and then converted back to the training
dtype. It is a floating-point approximation to the exact projection, not a proof
of rigorous rounding enclosures. Every call returns a traceable record with the
threshold convention, feasibility and distance correction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ram_experiments.config import feasibility_tolerance
from ram_experiments.energy import omega


@dataclass(frozen=True)
class ProjectionPolicy:
    """Numerical policy for the projection step."""

    device: str = "cpu"
    dtype: str = "float64"
    retry_on_violation: bool = True
    tolerance_mode: str = "primary"


def _project_singular_values(values: np.ndarray, radius: float) -> tuple[np.ndarray, float, str]:
    """Project non-negative singular values onto ``{z >= 0, sum(z) <= radius}``."""
    values = np.asarray(values, dtype=np.float64)
    if radius < 0.0:
        raise ValueError("tau must be non-negative")
    total = float(values.sum())
    if total <= radius:
        return values.copy(), 0.0, "interior"
    if radius == 0.0:
        threshold = float(values.max()) if values.size else 0.0
        return np.zeros_like(values), threshold, "zero_radius"

    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered)
    indices = np.arange(1, ordered.size + 1, dtype=np.float64)
    candidates = (cumulative - radius) / indices
    active = ordered > candidates
    j_star = 1 if not np.any(active) else int(np.nonzero(active)[0].max() + 1)
    threshold = float((cumulative[j_star - 1] - radius) / j_star)
    threshold = max(threshold, 0.0)
    projected = np.maximum(values - threshold, 0.0)
    return projected, threshold, "boundary"


def project_joint(
    weights: Sequence[np.ndarray],
    tau: float,
    *,
    policy: ProjectionPolicy | None = None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Project ``W`` onto ``C_tau = {W : 2 sum_l ||W_l||_* <= tau}``.

    The singular vectors are retained; only the concatenated singular values are
    projected. The output is a new list of arrays and never mutates the input.
    """
    policy = policy or ProjectionPolicy()
    if tau < 0:
        raise ValueError("tau must be non-negative")
    matrices = [np.asarray(weight, dtype=np.float64) for weight in weights]
    if not matrices:
        raise ValueError("at least one weight matrix is required")

    singular_values = []
    decompositions = []
    for matrix in matrices:
        q, sigma, vt = np.linalg.svd(matrix, full_matrices=False)
        decompositions.append((q, vt))
        singular_values.append(sigma)
    concatenated = np.concatenate(singular_values) if singular_values else np.zeros(0)
    radius = tau / 2.0
    projected_values, threshold, mode = _project_singular_values(concatenated, radius)

    if mode == "interior":
        projected = [matrix.copy() for matrix in matrices]
    else:
        projected = []
        offset = 0
        for (q, vt), sigma in zip(decompositions, singular_values, strict=True):
            size = sigma.size
            clipped = projected_values[offset : offset + size]
            offset += size
            projected.append((q * clipped) @ vt)

    omega_before = omega(matrices)
    omega_after = omega(projected)
    raw_violation = max(omega_after - tau, 0.0)
    tolerance = feasibility_tolerance(tau)
    record = {
        "tau": float(tau),
        "radius_tau_over_2": float(radius),
        "threshold_t": float(threshold),
        "threshold_convention": "t = 2*nu/rho for objective (1/2)||Z-Y||^2",
        "mode": mode,
        "omega_before": float(omega_before),
        "omega_after": float(omega_after),
        "raw_violation": float(raw_violation),
        "signed_slack": float(omega_after - tau),
        "relative_violation": float(raw_violation / max(1.0, abs(tau))),
        "feasible": bool(raw_violation <= tolerance),
        "tolerance": float(tolerance),
        "distance_frobenius": float(
            np.sqrt(sum(np.sum((a - b) ** 2) for a, b in zip(matrices, projected, strict=True)))
        ),
        "projection_device": policy.device,
        "projection_dtype": policy.dtype,
        "n_singular_values": int(concatenated.size),
    }
    return projected, record


def project_joint_torch(
    weights: Sequence[torch.Tensor],
    tau: float,
    *,
    policy: ProjectionPolicy | None = None,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """Torch wrapper around :func:`project_joint`.

    The SVD runs in ``float64`` on the declared projection device (CPU by default
    for numerical stability and predictable cost). The result is converted back
    to the original dtype and device.
    """
    policy = policy or ProjectionPolicy()
    references = list(weights)
    original_dtypes = [tensor.dtype for tensor in references]
    original_devices = [tensor.device for tensor in references]
    cpu_weights = [
        tensor.detach().to(device=policy.device, dtype=torch.float64).cpu().numpy()
        for tensor in references
    ]
    projected, record = project_joint(cpu_weights, tau, policy=policy)

    outputs = [
        torch.as_tensor(array, dtype=dtype, device=device)
        for array, dtype, device in zip(projected, original_dtypes, original_devices, strict=True)
    ]
    record["projection_device"] = policy.device
    record["projection_dtype"] = policy.dtype
    return outputs, record


def project_spectral(matrix: np.ndarray, radius: float) -> np.ndarray:
    """Frobenius projection onto ``||Z||_2 <= radius`` by clipping singular values."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    array = np.asarray(matrix, dtype=np.float64)
    q, sigma, vt = np.linalg.svd(array, full_matrices=False)
    return (q * np.minimum(sigma, radius)) @ vt


def project_layerwise(
    weights: Sequence[np.ndarray],
    radii: Sequence[float],
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Project each layer onto its own nuclear-norm ball ``||Z_l||_* <= r_l``."""
    matrices = [np.asarray(weight, dtype=np.float64) for weight in weights]
    if len(matrices) != len(radii):
        raise ValueError("one radius per layer is required")
    projected = []
    thresholds = []
    for matrix, radius in zip(matrices, radii, strict=True):
        if radius < 0:
            raise ValueError("radii must be non-negative")
        q, sigma, vt = np.linalg.svd(matrix, full_matrices=False)
        clipped, threshold, _ = _project_singular_values(sigma, radius)
        projected.append((q * clipped) @ vt)
        thresholds.append(float(threshold))
    total_after = float(sum(np.linalg.svd(z, compute_uv=False).sum() for z in projected))
    tau = 2.0 * float(sum(radii))
    raw_violation = max(2.0 * total_after - tau, 0.0)
    tolerance = 1e-6 + 1e-5 * max(1.0, abs(tau))
    record = {
        "radii": [float(r) for r in radii],
        "thresholds": thresholds,
        "omega_after": float(2.0 * total_after),
        "sum_radii": float(sum(radii)),
        "tau": tau,
        "raw_violation": float(raw_violation),
        "feasible": bool(raw_violation <= tolerance),
        "tolerance": float(tolerance),
        "mode": "layerwise",
    }
    return projected, record


def allocate_radii(
    reference_weights: Sequence[np.ndarray],
    tau: float,
    mode: str = "weights",
) -> list[float]:
    """Split ``tau/2`` across layers by parameter count or reference energy."""
    if tau < 0:
        raise ValueError("tau must be non-negative")
    matrices = [np.asarray(weight, dtype=np.float64) for weight in reference_weights]
    if not matrices:
        raise ValueError("at least one layer is required")
    if mode == "weights":
        scores = np.array([matrix.size for matrix in matrices], dtype=np.float64)
    elif mode == "reference":
        scores = np.array(
            [np.linalg.svd(matrix, compute_uv=False).sum() for matrix in matrices],
            dtype=np.float64,
        )
    elif mode == "uniform":
        scores = np.ones(len(matrices), dtype=np.float64)
    else:
        raise ValueError(f"unknown allocation mode: {mode}")
    total = float(scores.sum())
    proportions = np.full(len(matrices), 1.0 / len(matrices)) if total == 0.0 else scores / total
    return [float(value) for value in proportions * (tau / 2.0)]


def nuclear_norm_torch(matrix: torch.Tensor) -> torch.Tensor:
    """Differentiable nuclear norm for the NUC objective."""
    return torch.linalg.svdvals(matrix).sum()


def omega_torch(weights: Sequence[torch.Tensor]) -> torch.Tensor:
    """Differentiable ``Omega`` along the training path."""
    total = torch.zeros((), dtype=weights[0].dtype, device=weights[0].device)
    for weight in weights:
        total = total + nuclear_norm_torch(weight)
    return 2.0 * total
