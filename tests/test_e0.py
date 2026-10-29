"""E0: mathematical and numerical checks required before any training."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ram_experiments.config import ARCHITECTURES, TrainConfig, strict_tolerance
from ram_experiments.energy import (
    energy_ratio,
    graph_adjacency,
    graph_energy,
    nuclear_norm,
    omega,
    spectral_dimension,
)
from ram_experiments.methods import adapt_rho, budget_phase, initial_tau0
from ram_experiments.models import (
    apply_spectral_norm,
    effective_clone,
    effective_state_dict,
    make_model,
)
from ram_experiments.projection import (
    allocate_radii,
    omega_torch,
    project_joint,
    project_layerwise,
    project_spectral,
)


def _random_matrices(shapes, seed=0):
    generator = np.random.Generator(np.random.PCG64(seed))
    return [generator.normal(size=shape) for shape in shapes]


@pytest.mark.parametrize(
    "shapes",
    [
        [(1, 1)],
        [(2, 3)],
        [(3, 2)],
        [(4, 4)],
        [(3, 2), (2, 3)],
        [(4, 3), (3, 2), (2, 4), (4, 1)],
    ],
)
def test_joint_projection_shapes_and_radii(shapes):
    weights = _random_matrices(shapes, seed=sum(sum(shape) for shape in shapes))
    base = omega(weights)
    for factor in (0.0, 0.1, 0.8, 1.0, 1.2):
        tau = factor * base
        projected, record = project_joint(weights, tau)
        assert all(np.isfinite(layer).all() for layer in projected)
        assert record["omega_after"] <= tau + strict_tolerance(tau) * 10
        assert record["feasible"] or tau == 0.0
        if factor >= 1.0:
            assert np.allclose(projected[0], weights[0])
            assert record["mode"] == "interior"
        if factor == 0.0:
            assert all(np.allclose(layer, 0.0) for layer in projected)
        again, _ = project_joint(projected, tau)
        assert all(np.allclose(a, b, atol=1e-9) for a, b in zip(projected, again, strict=True))


def test_projection_is_nonexpansive_and_continuous_at_boundary():
    weights = _random_matrices([(4, 3), (3, 2)], seed=7)
    base = omega(weights)
    epsilon = 1e-7
    left, _ = project_joint(weights, base * (1 - epsilon))
    right, _ = project_joint(weights, base * (1 + epsilon))
    distance = math.sqrt(sum(np.sum((a - b) ** 2) for a, b in zip(left, right, strict=True)))
    assert distance < 1e-5


def test_repeated_spectrum_and_rotation_equivariance():
    matrix = np.diag([3.0, 3.0, 1.0, 0.0])
    tau = 0.8 * omega([matrix])
    projected, record = project_joint([matrix], tau)
    assert record["mode"] == "boundary"
    assert np.allclose(projected[0], projected[0].T)
    rotation, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(4, 4)))
    rotated, _ = project_joint([rotation @ matrix @ rotation.T], tau)
    assert np.allclose(rotated[0], rotation @ projected[0] @ rotation.T, atol=1e-9)


def test_zero_radius_returns_exact_zero():
    weights = _random_matrices([(3, 2), (2, 3)], seed=1)
    projected, record = project_joint(weights, 0.0)
    assert record["mode"] == "zero_radius"
    assert all(np.array_equal(layer, np.zeros_like(layer)) for layer in projected)


def test_spectral_projection_diag_2_1():
    projected = project_spectral(np.diag([2.0, 1.0]), 1.0)
    assert np.allclose(projected, np.diag([1.0, 1.0]))
    rescaling = np.diag([1.0, 0.5])
    assert np.sum((projected - np.diag([2.0, 1.0])) ** 2) == pytest.approx(1.0)
    assert np.sum((rescaling - np.diag([2.0, 1.0])) ** 2) == pytest.approx(1.25)
    assert np.allclose(project_spectral(np.zeros((3, 2)), 1.0), np.zeros((3, 2)))


def test_layerwise_projection_respects_radii():
    weights = _random_matrices([(3, 2), (2, 3)], seed=4)
    radii = [0.5, 1.5]
    projected, record = project_layerwise(weights, radii)
    for layer, radius in zip(projected, radii, strict=True):
        assert nuclear_norm(layer) <= radius + 1e-10
    assert record["sum_radii"] == pytest.approx(2.0)


def test_graph_energy_counterexample():
    adjacency = graph_adjacency([np.array([[1.0]]), np.array([[1.0]])])
    assert np.allclose(adjacency, [[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    assert graph_energy([np.array([[1.0]]), np.array([[1.0]])]) == pytest.approx(2 * math.sqrt(2))
    assert omega([np.array([[1.0]]), np.array([[1.0]])]) == pytest.approx(4.0)
    assert energy_ratio([np.array([[1.0]]), np.array([[1.0]])]) == pytest.approx(math.sqrt(2) / 2)


def test_energy_bounds_and_dimension_counts():
    weights = _random_matrices([(5, 3), (4, 5), (2, 4)], seed=5)
    ratio = energy_ratio(weights)
    assert ratio is not None and 0.5 - 1e-12 <= ratio <= 1.0 + 1e-12
    assert spectral_dimension(weights) == 3 + 4 + 2
    expected = {"3L": 394, "6L": 554, "9L": 730}
    for architecture, dimension in expected.items():
        widths = ARCHITECTURES[architecture]
        weights = [np.ones((widths[i + 1], widths[i])) for i in range(len(widths) - 1)]
        assert spectral_dimension(weights) == dimension


def test_factor_of_two_in_budget():
    weights = _random_matrices([(3, 2)], seed=9)
    projected, record = project_joint(weights, 2.0)
    nuclear_sum = sum(nuclear_norm(layer) for layer in projected)
    assert nuclear_sum <= 1.0 + 1e-10
    assert record["radius_tau_over_2"] == pytest.approx(1.0)


def test_rho_limits_preserve_scaled_dual():
    rho_next, rescale = adapt_rho(1_000.0, 1.0, 8.0)
    assert rho_next == pytest.approx(10.0)
    assert rho_next * rescale == pytest.approx(8.0)
    rho_next, rescale = adapt_rho(1.0, 1.0, 0.1)
    assert rho_next == pytest.approx(0.1)
    assert rescale == pytest.approx(1.0)
    rho_next, rescale = adapt_rho(0.0, 0.0, 2.0)
    assert rho_next == pytest.approx(2.0) and rescale == pytest.approx(1.0)


def test_budget_schedule_and_initial_tau0():
    config = TrainConfig()
    assert budget_phase(3, 10.0, 5.0, config).tau is None
    assert budget_phase(4, 10.0, 5.0, config).tau == pytest.approx(9.0)
    assert budget_phase(8, 10.0, 5.0, config).tau == pytest.approx(5.0)
    assert budget_phase(9, 10.0, 5.0, config).tau == pytest.approx(5.0)
    assert budget_phase(25, 10.0, 5.0, config).phase == "fixed"
    assert initial_tau0(12.0, 5.0) == pytest.approx(12.0)
    assert initial_tau0(2.0, 5.0) == pytest.approx(5.0)


def test_layer_allocation_sums_to_tau_over_two():
    reference = _random_matrices([(4, 3), (2, 4)], seed=12)
    for mode in ("weights", "reference", "uniform"):
        radii = allocate_radii(reference, 6.0, mode=mode)
        assert sum(radii) == pytest.approx(3.0)
    projected, record = project_layerwise(reference, allocate_radii(reference, 6.0, "weights"))
    assert record["feasible"]
    assert record["sum_radii"] == pytest.approx(3.0)
    assert all(np.isfinite(layer).all() for layer in projected)


def test_nuclear_norm_gradient_is_autograd_connected():
    diagonal = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64, requires_grad=True)
    matrix = torch.diag(diagonal)
    loss = omega_torch([matrix]) / 2.0
    loss.backward()
    assert diagonal.grad is not None
    assert torch.allclose(diagonal.grad, torch.ones(3, dtype=torch.float64), atol=1e-6)

    epsilon = 1e-6
    values = []
    for sign in (-1.0, 1.0):
        perturbed = torch.diag(torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64))
        perturbed[0, 0] += sign * epsilon
        values.append(float(torch.linalg.svdvals(perturbed).sum()))
    finite_difference = (values[1] - values[0]) / (2 * epsilon)
    assert finite_difference == pytest.approx(1.0, abs=1e-4)


def test_spectral_norm_exposes_effective_weights():
    model = make_model("3L", 42)
    original = [
        parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.endswith("weight")
    ]
    apply_spectral_norm(model, n_power_iterations=1, eps=1e-12)
    model.train()
    model(torch.randn(2, 784))
    effective = [layer.weight.detach() for layer in model.layers]
    for matrix in effective:
        spectral_norm = float(torch.linalg.matrix_norm(matrix, ord=2))
        # Power iteration is an estimate; the protocol requires measuring any excess.
        assert spectral_norm <= 1.05
    clone = effective_clone(model)
    model.eval()
    with torch.no_grad():
        inputs = torch.randn(3, 784)
        assert torch.allclose(model(inputs), clone(inputs), atol=1e-6)
    state = effective_state_dict(model)
    weight_keys = [key for key in state if key.endswith(".weight")]
    assert len(weight_keys) == len(effective)
    assert all("weight_orig" not in key for key in state)
    assert len(original) == len(effective)


def test_manifest_rejects_duplicate_configuration():
    from ram_experiments.manifest import generate_pilot_manifest, validate_manifest

    runs = generate_pilot_manifest(fingerprint="test")
    report = validate_manifest(runs)
    assert report["valid"]
    duplicated = [*runs, runs[0]]
    assert not validate_manifest(duplicated)["valid"]


def test_projection_oracle_cvxpy():
    cvxpy = pytest.importorskip("cvxpy")
    weights = _random_matrices([(3, 2), (2, 3)], seed=11)
    tau = 0.8 * omega(weights)
    projected, record = project_joint(weights, tau)
    variables = [cvxpy.Variable(weight.shape) for weight in weights]
    objective = 0.5 * sum(
        cvxpy.sum_squares(variable - weight)
        for variable, weight in zip(variables, weights, strict=True)
    )
    constraint = sum(cvxpy.normNuc(variable) for variable in variables) <= tau / 2.0
    problem = cvxpy.Problem(cvxpy.Minimize(objective), [constraint])
    try:
        problem.solve(
            solver="CLARABEL",
            tol_gap_abs=1e-10,
            tol_gap_rel=1e-10,
            tol_feas=1e-10,
            max_iter=200,
        )
    except (TypeError, cvxpy.error.SolverError):
        problem.solve(solver="CLARABEL", max_iter=200)
    assert problem.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}
    gap = sum(
        float(np.linalg.norm(variable.value - reference))
        for variable, reference in zip(variables, projected, strict=True)
    )
    assert gap <= 1e-6 * max(1.0, math.sqrt(sum(np.sum(weight**2) for weight in weights)))
    assert record["feasible"]
