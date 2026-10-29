"""E1: analytical quadratic-growth and strong-convexity checks."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ram_experiments.energy import omega
from ram_experiments.projection import project_joint


def _direction(shape, generator):
    value = generator.normal(size=shape)
    norm = np.linalg.norm(value)
    return value / norm if norm else value


def test_strong_convexity_ratio_bounded():
    gamma = 2.0
    dimension = 4
    weights_star = [np.zeros((3, 2)), np.zeros((2, 3))]
    omega_star = omega(weights_star)
    for seed in range(5):
        generator = np.random.Generator(np.random.PCG64(1000 + seed))
        for radius in (1e-8, 1e-6, 1e-4, 1e-2, 1.0):
            direction = _direction((3, 2), generator)
            perturbation = radius * direction
            weights = [weights_star[0] + perturbation, weights_star[1]]
            gap = 0.5 * gamma * np.sum(perturbation**2)
            left = abs(omega(weights) - omega_star)
            right = 2 * math.sqrt(2 * dimension / gamma) * math.sqrt(gap)
            assert left <= right + 1e-12


def test_qg_without_uniqueness_ratio_bounded():
    projection = np.diag([1.0, 0.0])
    target = np.array([[1.0, 2.0], [0.0, 0.0]])
    gamma = 3.0
    rank = 1
    generator = np.random.Generator(np.random.PCG64(20261013))
    for _ in range(200):
        weights = generator.normal(size=(2, 2))
        residual = projection @ (weights - target)
        gap = 0.5 * gamma * np.sum(residual**2)
        weights_plus = weights - residual
        left = abs(omega([weights]) - omega([weights_plus]))
        right = 2 * math.sqrt(2 * rank / gamma) * math.sqrt(gap)
        assert left <= right + 1e-12
        assert np.allclose(projection @ weights_plus, projection @ target)


def test_sharpness_case_ratio_is_one():
    t = 1.0
    weights = [np.eye(2) * t]
    gap = 0.5 * 1.0 * np.sum((np.eye(2) * t) ** 2)
    left = abs(omega(weights) - 0.0)
    right = 2 * math.sqrt(2 * 2 / 1.0) * math.sqrt(gap)
    assert left == pytest.approx(right)
    assert left / right == pytest.approx(1.0)


def test_projection_of_target_is_identity():
    target = np.array([[1.0, 2.0], [0.0, 0.0]])
    tau = 10.0 * omega([target])
    projected, record = project_joint([target], tau)
    assert record["mode"] == "interior"
    assert np.allclose(projected[0], target)
