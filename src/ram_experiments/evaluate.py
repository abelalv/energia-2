"""Checkpoint evaluation, spectra and feasibility certificates."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ram_experiments.energy import (
    graph_energy,
    omega,
    perturbation_bound,
    spectral_dimension,
    spectral_summary,
    violation,
    weights_sha256,
)
from ram_experiments.models import MLP


@torch.no_grad()
def evaluate_model(
    model: MLP,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_size: int = 1024,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Mean cross-entropy and accuracy for a model on a fixed tensor split."""
    model = model.to(device)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = int(targets.numel())
    for start in range(0, total, batch_size):
        batch_x = inputs[start : start + batch_size].to(device)
        batch_y = targets[start : start + batch_size].to(device)
        logits = model(batch_x)
        loss = torch.nn.functional.cross_entropy(logits, batch_y, reduction="sum")
        total_loss += float(loss.detach().cpu())
        total_correct += int((logits.argmax(dim=1) == batch_y).sum().cpu())
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": total_correct / max(total, 1),
        "n_examples": total,
    }


def spectral_report(weights: list[torch.Tensor] | list[np.ndarray]) -> dict[str, Any]:
    """Per-layer spectra and global Omega / graph energy."""
    matrices = [
        weight.detach().cpu().numpy() if isinstance(weight, torch.Tensor) else np.asarray(weight)
        for weight in weights
    ]
    layers = [spectral_summary(matrix) for matrix in matrices]
    return {
        "omega": omega(matrices),
        "graph_energy": graph_energy(matrices),
        "energy_ratio": (
            None if omega(matrices) == 0.0 else graph_energy(matrices) / omega(matrices)
        ),
        "D": spectral_dimension(matrices),
        "weights_sha256": weights_sha256(matrices),
        "layers": layers,
    }


def feasibility_report(
    weights: list[torch.Tensor] | list[np.ndarray],
    tau: float,
    z: list[torch.Tensor] | list[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Direct violation and the two dimension-based certificates."""
    matrices = [
        weight.detach().cpu().numpy() if isinstance(weight, torch.Tensor) else np.asarray(weight)
        for weight in weights
    ]
    direct = violation(matrices, tau)
    report: dict[str, Any] = {
        "tau": float(tau),
        "omega": omega(matrices),
        "violation": float(direct),
        "feasible": bool(direct <= 1e-6 + 1e-5 * max(1.0, abs(tau))),
    }
    if z is not None:
        z_matrices = [
            layer.detach().cpu().numpy() if isinstance(layer, torch.Tensor) else np.asarray(layer)
            for layer in z
        ]
        z_violation = violation(z_matrices, tau)
        nuclear_bound = z_violation + 2.0 * sum(
            np.linalg.svd(a - b, compute_uv=False).sum()
            for a, b in zip(matrices, z_matrices, strict=True)
        )
        dimension_bound = z_violation + perturbation_bound(
            matrices, z_matrices, dimension=spectral_dimension(matrices)
        )
        report.update(
            {
                "z_violation": float(z_violation),
                "nuclear_bound": float(nuclear_bound),
                "dimension_bound": float(dimension_bound),
                "nuclear_bound_minus_direct": float(nuclear_bound - direct),
                "dimension_bound_minus_direct": float(dimension_bound - direct),
            }
        )
    return report


def evaluation_record(
    *,
    variant: str,
    model: MLP,
    bundle: Any,
    tau_final: float | None,
    device: torch.device | str,
    allow_test: bool = False,
) -> dict[str, Any]:
    """Metrics for one model variant (W, Z or projected W)."""
    weights = [layer.weight.detach().clone() for layer in model.layers]
    record: dict[str, Any] = {
        "variant": variant,
        "train": evaluate_model(model, bundle.train_x, bundle.train_y, device=device),
        "validation": evaluate_model(model, bundle.val_x, bundle.val_y, device=device),
        "spectral": spectral_report(weights),
        "feasibility": (
            feasibility_report(weights, tau_final)
            if tau_final is not None
            else {"tau": None, "applicable": False, "reason": "unconstrained method"}
        ),
    }
    if allow_test:
        record["test"] = evaluate_model(model, bundle.test_x, bundle.test_y, device=device)
    return record
