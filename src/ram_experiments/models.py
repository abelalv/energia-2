"""ReLU MLPs used by the protocol and helpers to snapshot effective weights."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Sequence

import torch
from torch import nn

from ram_experiments.config import ARCHITECTURES
from ram_experiments.energy import dimension_counts


class MLP(nn.Module):
    """Fully connected ReLU network with a linear output layer."""

    def __init__(self, dims: Sequence[int]) -> None:
        super().__init__()
        if len(dims) < 2:
            raise ValueError("an MLP needs at least one weight matrix")
        self.dims = tuple(int(value) for value in dims)
        self.layers = nn.ModuleList(
            nn.Linear(self.dims[index], self.dims[index + 1]) for index in range(len(self.dims) - 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        activations = inputs
        for index, layer in enumerate(self.layers):
            activations = layer(activations)
            if index < len(self.layers) - 1:
                activations = torch.relu(activations)
        return activations

    def linear_layers(self) -> list[nn.Linear]:
        return list(self.layers)

    def weight_matrices(self) -> list[torch.Tensor]:
        """Effective weights: ``module.weight`` resolves spectral parametrization."""
        return [layer.weight for layer in self.layers]

    def bias_vectors(self) -> list[torch.Tensor]:
        return [layer.bias for layer in self.layers if layer.bias is not None]


def make_model(architecture: str, seed: int) -> MLP:
    """Create an MLP with the protocol architecture and deterministic init."""
    if architecture not in ARCHITECTURES:
        raise KeyError(f"unknown architecture {architecture!r}")
    torch.manual_seed(seed)
    return MLP(ARCHITECTURES[architecture])


def initial_state(model: MLP) -> OrderedDict[str, torch.Tensor]:
    """Immutable copy of the initialization, taken before SN parametrization."""
    return OrderedDict(
        (name, tensor.detach().clone()) for name, tensor in model.state_dict().items()
    )


def effective_state_dict(model: MLP) -> OrderedDict[str, torch.Tensor]:
    """State dict with effective ``weight`` entries, independent of parametrization."""
    state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for index, layer in enumerate(model.layers):
        state[f"layers.{index}.weight"] = layer.weight.detach().clone()
        if layer.bias is not None:
            state[f"layers.{index}.bias"] = layer.bias.detach().clone()
    return state


def effective_clone(model: MLP) -> MLP:
    """Plain MLP containing the current effective weights and the same biases."""
    clone = MLP(model.dims)
    clone.load_state_dict(effective_state_dict(model))
    clone.eval()
    return clone


def apply_spectral_norm(model: MLP, n_power_iterations: int = 1, eps: float = 1e-12) -> MLP:
    """Apply PyTorch's spectral_norm parametrization to every linear weight."""
    for layer in model.layers:
        torch.nn.utils.parametrizations.spectral_norm(
            layer,
            name="weight",
            n_power_iterations=n_power_iterations,
            eps=eps,
        )
    return model


def effective_weights(model: MLP) -> list[torch.Tensor]:
    """Detached effective weight matrices (works with and without SN)."""
    return [layer.weight.detach().clone() for layer in model.layers]


def parameters_fingerprint(model: MLP) -> str:
    """SHA-256 of effective weights and biases."""
    digest = hashlib.sha256()
    for layer in model.layers:
        for tensor in (
            layer.weight.detach(),
            layer.bias.detach() if layer.bias is not None else None,
        ):
            if tensor is None:
                continue
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def architecture_counts(architecture: str) -> dict[str, int]:
    """Weight/parameter/D counts for a named architecture."""
    return dimension_counts(ARCHITECTURES[architecture])


def count_parameters(model: nn.Module) -> int:
    """Number of scalar parameters, including biases and SN originals."""
    return int(sum(parameter.numel() for parameter in model.parameters()))


def count_effective_weights(model: MLP) -> int:
    """Number of scalar weights in the effective network."""
    return int(sum(layer.weight.numel() for layer in model.layers))
