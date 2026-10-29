"""Frozen run inventories, deduplication and sharding.

The main inventory follows the protocol exactly: 540 E2 runs (including the 300
WD/NUC calibration candidates), 60 E4 allocations and 10 long-horizon
continuations. Pilot runs are a separate five-epoch inventory.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ram_experiments.config import (
    ALPHAS,
    ARCHITECTURES,
    DATASETS,
    NUC_GRID,
    PILOT_EPOCHS,
    PILOT_SEED,
    PROTOCOL_VERSION,
    SEEDS,
    WD_GRID,
    RunConfig,
    TrainConfig,
    code_fingerprint,
)


def _base_config(
    *,
    stage: str,
    dataset: str,
    architecture: str,
    seed: int,
    method: str,
    alpha: float | None = None,
    reg_lambda: float | None = None,
    allocation: str | None = None,
    parent_run_id: str | None = None,
    tau_source: str | None = None,
    epochs: int = 25,
    fingerprint: str = "unknown",
) -> RunConfig:
    return RunConfig(
        protocol=PROTOCOL_VERSION,
        stage=stage,
        dataset=dataset,
        architecture=architecture,
        seed=seed,
        method=method,
        alpha=alpha,
        reg_lambda=reg_lambda,
        allocation=allocation,
        parent_run_id=parent_run_id,
        tau_source=tau_source,
        train=TrainConfig(epochs=epochs),
        code_fingerprint=fingerprint,
        numerical_profile="float32-train_float64-cpu-projection",
    )


def generate_pilot_manifest(fingerprint: str | None = None) -> list[RunConfig]:
    """36 pilot runs: both datasets, three architectures, six methods, seed 42."""
    fingerprint = fingerprint or code_fingerprint()
    runs: list[RunConfig] = []
    for dataset in DATASETS:
        for architecture in ARCHITECTURES:
            base = {
                "stage": "pilot",
                "dataset": dataset,
                "architecture": architecture,
                "seed": PILOT_SEED,
                "epochs": PILOT_EPOCHS,
                "fingerprint": fingerprint,
            }
            runs.append(_base_config(method="STD", tau_source=None, **base))
            runs.append(_base_config(method="WD", reg_lambda=1e-3, tau_source="pilot_std", **base))
            runs.append(_base_config(method="NUC", reg_lambda=1e-3, tau_source="pilot_std", **base))
            runs.append(_base_config(method="SN", tau_source="pilot_std", **base))
            runs.append(_base_config(method="ADMM", alpha=0.8, tau_source="pilot_std", **base))
            runs.append(_base_config(method="PSGD-M", alpha=0.8, tau_source="pilot_std", **base))
    return runs


def generate_main_manifest(fingerprint: str | None = None) -> list[RunConfig]:
    """Frozen main inventory: 540 E2 + 60 E4 + 10 continuations."""
    fingerprint = fingerprint or code_fingerprint()
    runs: list[RunConfig] = []
    e2_lookup: dict[tuple[str, str, int, float], str] = {}

    for dataset in DATASETS:
        for architecture in ARCHITECTURES:
            for seed in SEEDS:
                base = {
                    "stage": "E2",
                    "dataset": dataset,
                    "architecture": architecture,
                    "seed": seed,
                    "epochs": 25,
                    "fingerprint": fingerprint,
                }
                runs.append(_base_config(method="STD", tau_source=None, **base))
                runs.append(_base_config(method="SN", tau_source="std_ref", **base))
                for coefficient in WD_GRID:
                    runs.append(
                        _base_config(
                            method="WD", reg_lambda=coefficient, tau_source="std_ref", **base
                        )
                    )
                for coefficient in NUC_GRID:
                    runs.append(
                        _base_config(
                            method="NUC", reg_lambda=coefficient, tau_source="std_ref", **base
                        )
                    )
                for alpha in ALPHAS:
                    admm = _base_config(method="ADMM", alpha=alpha, tau_source="std_ref", **base)
                    runs.append(admm)
                    e2_lookup[(dataset, architecture, seed, alpha)] = admm.run_id()
                    runs.append(
                        _base_config(method="PSGD-M", alpha=alpha, tau_source="std_ref", **base)
                    )

    for dataset in DATASETS:
        for architecture in ARCHITECTURES:
            for seed in SEEDS:
                for allocation in ("weights", "reference"):
                    runs.append(
                        _base_config(
                            stage="E4",
                            dataset=dataset,
                            architecture=architecture,
                            seed=seed,
                            method="ADMM",
                            alpha=0.8,
                            allocation=allocation,
                            tau_source="std_ref",
                            epochs=25,
                            fingerprint=fingerprint,
                        )
                    )

    for dataset in DATASETS:
        for seed in SEEDS:
            parent = e2_lookup[(dataset, "9L", seed, 0.8)]
            runs.append(
                _base_config(
                    stage="E4",
                    dataset=dataset,
                    architecture="9L",
                    seed=seed,
                    method="ADMM",
                    alpha=0.8,
                    tau_source="parent",
                    parent_run_id=parent,
                    epochs=100,
                    fingerprint=fingerprint,
                )
            )
    return runs


def write_manifest(runs: list[RunConfig], path: str | Path) -> Path:
    """Write one JSON object per line, sorted by run_id."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for run in sorted(runs, key=lambda item: item.run_id()):
            handle.write(json.dumps(run.to_dict(), sort_keys=True) + "\n")
    return path


def read_manifest(path: str | Path) -> list[RunConfig]:
    path = Path(path)
    runs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            runs.append(RunConfig.from_dict(json.loads(line)))
    return runs


def validate_manifest(runs: list[RunConfig]) -> dict[str, Any]:
    """Detect duplicate scientific identities and summarize the inventory."""
    keys: Counter[str] = Counter()
    ids: Counter[str] = Counter()
    for run in runs:
        keys[run.config_hash()] += 1
        ids[run.run_id()] += 1
    duplicates = [{"config_hash": key, "count": count} for key, count in keys.items() if count > 1]
    duplicate_ids = [run_id for run_id, count in ids.items() if count > 1]
    return {
        "n_runs": len(runs),
        "n_unique_configurations": len(keys),
        "n_unique_run_ids": len(ids),
        "duplicates": duplicates,
        "duplicate_run_ids": duplicate_ids,
        "valid": not duplicates and not duplicate_ids,
        "by_stage": dict(Counter(run.stage for run in runs)),
        "by_method": dict(Counter(run.method for run in runs)),
        "by_dataset": dict(Counter(run.dataset for run in runs)),
        "by_architecture": dict(Counter(run.architecture for run in runs)),
        "by_alpha": dict(Counter(str(run.alpha) for run in runs)),
        "total_epochs": int(sum(run.train.epochs for run in runs)),
    }


def shard_manifest(runs: list[RunConfig], shard: int, total_shards: int) -> list[RunConfig]:
    """Deterministic round-robin shard of a manifest."""
    if total_shards < 1:
        raise ValueError("total_shards must be >= 1")
    if not 0 <= shard < total_shards:
        raise ValueError("shard must satisfy 0 <= shard < total_shards")
    ordered = sorted(runs, key=lambda item: item.run_id())
    return [run for index, run in enumerate(ordered) if index % total_shards == shard]


def expected_counts() -> dict[str, int]:
    """Protocol counts used by tests and the audit command."""
    return {
        "pilot": 2 * 3 * 6,
        "E2_core": 30 + 90 + 90 + 150 + 150 + 30,
        "E4_allocations": 2 * 3 * 5 * 2,
        "E4_continuations": 2 * 5,
        "main_total": 540 + 60 + 10,
    }


@dataclass(frozen=True)
class InventoryReport:
    """Convenience wrapper for JSON printing."""

    payload: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(self.payload, indent=2, sort_keys=True)
