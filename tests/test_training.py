"""Inventory, data-split and training smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ram_experiments.config import RunConfig, TrainConfig
from ram_experiments.data import (
    build_bundles,
    load_bundle,
    stratified_prefix,
    stratified_split,
    synthetic_bundle,
)
from ram_experiments.manifest import (
    expected_counts,
    generate_main_manifest,
    generate_pilot_manifest,
    read_manifest,
    validate_manifest,
    write_manifest,
)
from ram_experiments.train import train_run


def test_pilot_inventory_has_36_runs():
    runs = generate_pilot_manifest(fingerprint="test")
    report = validate_manifest(runs)
    assert report["n_runs"] == 36
    assert report["valid"]
    assert report["by_method"] == {"ADMM": 6, "NUC": 6, "PSGD-M": 6, "SN": 6, "STD": 6, "WD": 6}


def test_main_inventory_has_610_runs_and_no_duplicates():
    runs = generate_main_manifest(fingerprint="test")
    report = validate_manifest(runs)
    assert report["n_runs"] == 610
    assert report["valid"]
    assert report["by_stage"] == {"E2": 540, "E4": 70}
    assert report["total_epochs"] == 540 * 25 + 60 * 25 + 10 * 100
    assert expected_counts()["main_total"] == 610


def test_manifest_roundtrip(tmp_path: Path):
    runs = generate_pilot_manifest(fingerprint="test")
    path = write_manifest(runs, tmp_path / "pilot.jsonl")
    loaded = read_manifest(path)
    assert len(loaded) == 36
    assert {run.config_hash() for run in loaded} == {run.config_hash() for run in runs}
    assert loaded[0].train.save_epochs == (0, 3, 8, 15, 25)


def test_alpha_independent_methods_do_not_duplicate():
    runs = generate_main_manifest(fingerprint="test")
    wd = [run for run in runs if run.method == "WD"]
    assert len(wd) == 150
    assert all(run.alpha is None for run in wd)
    admm = [run for run in runs if run.method == "ADMM" and run.stage == "E2"]
    assert len(admm) == 90
    assert {run.alpha for run in admm} == {0.6, 0.8, 1.0}


def test_stratified_split_counts():
    labels = np.repeat(np.arange(10), 6000)
    train_idx, val_idx = stratified_split(labels, total_validation=10_000, seed=20261013)
    assert train_idx.size == 50_000
    assert val_idx.size == 10_000
    for label in range(10):
        assert np.sum(labels[val_idx] == label) == 1000
    assert len(np.intersect1d(train_idx, val_idx)) == 0


def test_pilot_prefix_is_stratified():
    labels = np.repeat(np.arange(10), 6000)
    train_idx, val_idx = stratified_split(labels, total_validation=10_000, seed=20261013)
    pilot_train = stratified_prefix(train_idx, labels, 10_000)
    pilot_val = stratified_prefix(val_idx, labels, 2_000)
    assert pilot_train.size == 10_000
    assert pilot_val.size == 2_000
    assert np.all(np.bincount(labels[pilot_train], minlength=10) == 1000)
    assert np.all(np.bincount(labels[pilot_val], minlength=10) == 200)


def test_build_bundles_and_normalization():
    generator = np.random.default_rng(0)
    images = generator.integers(0, 256, size=(60_000, 784), dtype=np.uint8)
    labels = np.repeat(np.arange(10), 6000)
    test_images = generator.integers(0, 256, size=(10_000, 784), dtype=np.uint8)
    test_labels = np.repeat(np.arange(10), 1000)
    bundles = build_bundles("mnist", images, labels, test_images, test_labels)
    assert bundles["main"].train_x.shape == (50_000, 784)
    assert bundles["main"].val_x.shape == (10_000, 784)
    assert bundles["pilot"].train_x.shape == (10_000, 784)
    assert bundles["pilot"].val_x.shape == (2_000, 784)
    assert bundles["main"].test_x.shape == (10_000, 784)
    assert bundles["main"].std > 0


def test_prepare_and_load_synthetic_npz(tmp_path: Path):
    bundle = synthetic_bundle(n_train=64, n_val=16, n_test=16, input_dim=784)
    path = tmp_path / "mnist_main.npz"
    np.savez_compressed(
        path,
        train_x=bundle.train_x.numpy(),
        train_y=bundle.train_y.numpy(),
        val_x=bundle.val_x.numpy(),
        val_y=bundle.val_y.numpy(),
        test_x=bundle.test_x.numpy(),
        test_y=bundle.test_y.numpy(),
        mean=np.asarray(bundle.mean),
        std=np.asarray(bundle.std),
        indices_train=np.arange(64),
        indices_val=np.arange(16),
        hashes=json.dumps({}),
        name="synthetic",
        split="main",
    )
    loaded = load_bundle("mnist", split="main", processed_dir=tmp_path)
    assert loaded.train_x.shape[0] == 64


def _run_config(method: str, epochs: int, tau_source: str | None = None) -> RunConfig:
    return RunConfig(
        protocol="RAM-2026-10-13-v1",
        stage="test",
        dataset="mnist",
        architecture="3L",
        seed=42,
        method=method,
        train=TrainConfig(epochs=epochs, batch_size=8),
        code_fingerprint="test",
        tau_source=tau_source,
    )


def test_std_training_smoke(tmp_path: Path):
    bundle = synthetic_bundle(n_train=32, n_val=16, n_test=16, input_dim=784, seed=1)
    result = train_run(
        bundle,
        _run_config("STD", epochs=1),
        tmp_path / "std",
        device="cpu",
        allow_test=False,
    )
    assert result.status == "completed"
    assert result.epochs_completed == 1
    assert (tmp_path / "std" / "final_evaluation.json").exists()
    assert (tmp_path / "std" / "checkpoint.pt").exists()
    metrics = [
        json.loads(line) for line in (tmp_path / "std" / "epochs.jsonl").read_text().splitlines()
    ]
    assert np.isfinite(metrics[0]["train_loss"])
    assert np.isfinite(metrics[0]["omega_W"])
    assert metrics[0]["train_data_loss"] == pytest.approx(metrics[0]["train_loss"])
    assert metrics[0]["train_reg_term"] == 0.0
    assert metrics[0]["train_aug_term"] == 0.0


def test_psgd_m_projection_smoke(tmp_path: Path):
    bundle = synthetic_bundle(n_train=32, n_val=16, n_test=16, input_dim=784, seed=2)
    tau_final = 0.8 * 2.0 * 784  # a deliberately loose budget
    result = train_run(
        bundle,
        _run_config("PSGD-M", epochs=4),
        tmp_path / "psgd",
        device="cpu",
        tau_final=tau_final,
    )
    assert result.status == "completed"
    payload = json.loads((tmp_path / "psgd" / "final_evaluation.json").read_text())
    assert payload["W"]["feasibility"]["feasible"]


def test_admm_smoke(tmp_path: Path):
    bundle = synthetic_bundle(n_train=32, n_val=16, n_test=16, input_dim=784, seed=3)
    tau_final = 1.5 * 2.0 * 784
    result = train_run(
        bundle,
        _run_config("ADMM", epochs=4),
        tmp_path / "admm",
        device="cpu",
        tau_final=tau_final,
    )
    assert result.status == "completed"
    payload = json.loads((tmp_path / "admm" / "final_evaluation.json").read_text())
    assert "Z" in payload
    assert payload["Z"]["feasibility"]["feasible"]


def test_resume_matches_continuous_run(tmp_path: Path):
    bundle = synthetic_bundle(n_train=32, n_val=16, n_test=16, input_dim=784, seed=4)
    train_run(
        bundle,
        _run_config("STD", epochs=2),
        tmp_path / "continuous",
        device="cpu",
    )
    first = train_run(
        bundle,
        _run_config("STD", epochs=1),
        tmp_path / "resumed",
        device="cpu",
    )
    assert first.status == "completed"
    resumed = train_run(
        bundle,
        _run_config("STD", epochs=2),
        tmp_path / "resumed",
        device="cpu",
        resume=True,
    )
    assert resumed.status == "completed"
    continuous_state = torch.load(tmp_path / "continuous" / "final.pt", weights_only=False)["W"]
    resumed_state = torch.load(tmp_path / "resumed" / "final.pt", weights_only=False)["W"]
    for key in continuous_state:
        assert torch.allclose(continuous_state[key], resumed_state[key], atol=1e-6)
