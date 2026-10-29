"""Objectives, budget schedule and dual adaptation rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional

from ram_experiments.config import TrainConfig
from ram_experiments.projection import nuclear_norm_torch


def data_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over the minibatch."""
    return functional.cross_entropy(logits, targets)


def weight_decay_penalty(model: torch.nn.Module, coefficient: float) -> torch.Tensor:
    """``(lambda/2) ||theta||^2`` over weights and biases."""
    total = None
    for parameter in model.parameters():
        term = torch.sum(parameter**2)
        total = term if total is None else total + term
    if total is None:
        raise ValueError("model has no parameters")
    return 0.5 * coefficient * total


def nuclear_penalty(weights: list[torch.Tensor], coefficient: float) -> torch.Tensor:
    """``lambda * sum_l ||W_l||_*`` retaining autograd; biases excluded."""
    total = None
    for weight in weights:
        term = nuclear_norm_torch(weight)
        total = term if total is None else total + term
    if total is None:
        raise ValueError("at least one weight matrix is required")
    return coefficient * total


def augmented_penalty(
    weights: list[torch.Tensor],
    z: list[torch.Tensor],
    u: list[torch.Tensor],
    rho: float,
) -> torch.Tensor:
    """ADMM augmented term ``(rho/2) ||W - Z + u||_F^2`` over weights only."""
    total = None
    for weight, z_layer, u_layer in zip(weights, z, u, strict=True):
        term = torch.sum((weight - z_layer + u_layer) ** 2)
        total = term if total is None else total + term
    if total is None:
        raise ValueError("at least one weight matrix is required")
    return 0.5 * rho * total


def objective_term(
    method: str,
    model: torch.nn.Module,
    weights: list[torch.Tensor],
    coefficient: float | None,
) -> torch.Tensor | None:
    """Regularization term for the penalized baselines."""
    if coefficient is None:
        return None
    if method == "WD":
        return weight_decay_penalty(model, coefficient)
    if method == "NUC":
        return nuclear_penalty(weights, coefficient)
    return None


@dataclass(frozen=True)
class BudgetPhase:
    """One epoch of the budget schedule."""

    epoch: int
    phase: str
    tau: float | None
    j: int | None = None


def budget_phase(
    epoch: int, tau0: float | None, tau_final: float | None, config: TrainConfig
) -> BudgetPhase:
    """Return the phase and budget for epoch ``epoch`` (1-based)."""
    warmup = config.warmup_epochs
    transition = config.transition_epochs
    if epoch <= warmup:
        return BudgetPhase(epoch=epoch, phase="warmup", tau=None)
    if epoch <= warmup + transition:
        if tau0 is None or tau_final is None:
            raise ValueError("tau0 and tau_final are required after warmup")
        j = epoch - warmup
        tau = (1.0 - j / transition) * tau0 + (j / transition) * tau_final
        return BudgetPhase(epoch=epoch, phase="transition", tau=float(tau), j=j)
    if tau_final is None:
        raise ValueError("tau_final is required in the fixed phase")
    return BudgetPhase(epoch=epoch, phase="fixed", tau=float(tau_final))


def initial_tau0(omega_at_epoch3: float, tau_final: float) -> float:
    """Freeze ``tau0 = max(Omega(W^3), tau_final)`` at the end of warmup."""
    return float(max(omega_at_epoch3, tau_final))


def adapt_rho(
    residual_norm: float,
    dual_residual_norm: float,
    rho: float,
    *,
    balance: float = 10.0,
    factor: float = 2.0,
    rho_min: float = 0.1,
    rho_max: float = 10.0,
) -> tuple[float, float]:
    """ADMM residual balancing with clipping and dual rescaling factor.

    Returns ``(rho_next, dual_rescale_factor)`` where the caller must apply
    ``u <- (rho_old / rho_new) * u`` with the actual post-clipping values.
    """
    if residual_norm == 0.0 and dual_residual_norm == 0.0:
        return float(rho), 1.0
    candidate = rho
    if residual_norm > balance * dual_residual_norm:
        candidate = rho * factor
    elif dual_residual_norm > balance * residual_norm:
        candidate = rho / factor
    rho_next = float(min(max(candidate, rho_min), rho_max))
    rescale = float(rho / rho_next)
    return rho_next, rescale


def residual_norms(
    weights: list[torch.Tensor],
    z: list[torch.Tensor],
    z_old: list[torch.Tensor],
    rho: float,
) -> tuple[float, float]:
    """ADMM primal ``||W-Z||_F`` and dual ``rho ||Z-Z_old||_F`` residuals."""
    primal = torch.sqrt(
        sum(torch.sum((w - z_layer) ** 2) for w, z_layer in zip(weights, z, strict=True))
    )
    dual = rho * torch.sqrt(
        sum(torch.sum((z_layer - old) ** 2) for z_layer, old in zip(z, z_old, strict=True))
    )
    return float(primal), float(dual)


def optimizer_for(model: torch.nn.Module, config: TrainConfig) -> torch.optim.SGD:
    """Protocol SGD: constant LR, momentum, no weight decay, no Nesterov."""
    return torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        dampening=config.dampening,
        weight_decay=config.weight_decay,
        nesterov=config.nesterov,
    )


def method_metadata(method: str) -> dict[str, Any]:
    """Stable labels written to every loss/metric record."""
    mapping = {
        "STD": {"objective_id": "f_data", "regularization": None},
        "WD": {"objective_id": "f_data + (lambda/2)||theta||^2", "regularization": "weight_decay"},
        "NUC": {"objective_id": "f_data + lambda*sum||W||_*", "regularization": "nuclear"},
        "SN": {
            "objective_id": "f_data (spectral parametrization)",
            "regularization": "spectral_norm",
        },
        "ADMM": {"objective_id": "f_data + I_Ctau(Z) + augmented", "regularization": "admm"},
        "PSGD-M": {"objective_id": "f_data with projected SGD-M", "regularization": "projection"},
    }
    if method not in mapping:
        raise KeyError(f"unknown method {method!r}")
    return {"method": method, **mapping[method]}
