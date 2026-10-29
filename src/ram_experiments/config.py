"""Protocol constants and canonical run configurations.

A configuration is the unit of scientific identity. Two runs with the same
``config_hash`` are the same experiment and must not be trained twice: the
runner reuses the existing artifact instead. ``attempt_id`` is deliberately
excluded from the hash.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "RAM-2026-10-13-v1"

ARCHITECTURES: dict[str, tuple[int, ...]] = {
    "3L": (784, 256, 128, 10),
    "6L": (784, 256, 128, 64, 64, 32, 10),
    "9L": (784, 256, 128, 128, 64, 64, 32, 32, 16, 10),
}

DATASETS: tuple[str, ...] = ("mnist", "fashion_mnist")

SEEDS: tuple[int, ...] = (42, 123, 456, 789, 1024)

ALPHAS: tuple[float, ...] = (0.6, 0.8, 1.0)

METHODS: tuple[str, ...] = ("STD", "WD", "NUC", "SN", "ADMM", "PSGD-M")

#: Methods whose training problem does not depend on the budget factor.
ALPHA_INDEPENDENT_METHODS: frozenset[str] = frozenset({"STD", "WD", "NUC", "SN"})

WD_GRID: tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
NUC_GRID: tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)

PILOT_SEED = 42
PILOT_EPOCHS = 5
PILOT_TRAIN_SIZE = 10_000
PILOT_VAL_SIZE = 2_000
MAIN_TRAIN_SIZE = 50_000
MAIN_VAL_SIZE = 10_000

SPLIT_SEED = 20261013


@dataclass(frozen=True)
class TrainConfig:
    """Fixed hyperparameters of the protocol (section 4 of the spec)."""

    epochs: int = 25
    batch_size: int = 64
    lr: float = 0.01
    momentum: float = 0.9
    dampening: float = 0.0
    nesterov: bool = False
    weight_decay: float = 0.0
    warmup_epochs: int = 3
    transition_epochs: int = 5
    admm_rho: float = 1.0
    rho_min: float = 0.1
    rho_max: float = 10.0
    rho_balance: float = 10.0
    rho_factor: float = 2.0
    sn_n_power_iterations: int = 1
    sn_eps: float = 1e-12
    eval_batch_size: int = 1024
    precision: str = "float32"
    tf32: bool = False
    amp: bool = False
    projection_dtype: str = "float64"
    projection_device: str = "cpu"
    projection_frequency: str = "minibatch_plus_epoch_entry"
    save_epochs: tuple[int, ...] = (0, 3, 8, 15, 25)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunConfig:
    """One scientific run (a row of the frozen manifest)."""

    protocol: str
    stage: str
    dataset: str
    architecture: str
    seed: int
    method: str
    alpha: float | None = None
    reg_lambda: float | None = None
    allocation: str | None = None
    parent_run_id: str | None = None
    tau_source: str | None = None
    train: TrainConfig = field(default_factory=TrainConfig)
    code_fingerprint: str = "unknown"
    numerical_profile: str = "cpu-float64-projection"

    @property
    def dims(self) -> tuple[int, ...]:
        return ARCHITECTURES[self.architecture]

    @property
    def uses_budget(self) -> bool:
        return self.method in {"ADMM", "PSGD-M"}

    @property
    def depends_on_alpha(self) -> bool:
        return self.method not in ALPHA_INDEPENDENT_METHODS

    def scientific_key(self) -> dict[str, Any]:
        """Fields that define the experiment, excluding attempts and paths."""
        return {
            "protocol": self.protocol,
            "stage": self.stage,
            "dataset": self.dataset,
            "architecture": self.architecture,
            "seed": self.seed,
            "method": self.method,
            "alpha": self.alpha,
            "reg_lambda": self.reg_lambda,
            "allocation": self.allocation,
            "parent_run_id": self.parent_run_id,
            "tau_source": self.tau_source,
            "train": self.train.as_dict(),
            "code_fingerprint": self.code_fingerprint,
            "numerical_profile": self.numerical_profile,
        }

    def canonical_json(self) -> str:
        return json.dumps(self.scientific_key(), sort_keys=True, separators=(",", ":"))

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def run_id(self) -> str:
        parts = [
            self.stage,
            self.dataset,
            self.architecture,
            self.method,
            f"s{self.seed}",
        ]
        if self.alpha is not None:
            parts.append(f"a{self.alpha:g}")
        if self.reg_lambda is not None:
            parts.append(f"l{self.reg_lambda:g}")
        if self.allocation is not None:
            parts.append(self.allocation)
        if self.parent_run_id is not None:
            parts.append(f"cont-{self.parent_run_id[:8]}")
        parts.append(self.config_hash()[:16])
        return "-".join(parts)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["train"] = self.train.as_dict()
        payload["run_id"] = self.run_id()
        payload["config_hash"] = self.config_hash()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunConfig:
        data = dict(payload)
        data.pop("run_id", None)
        data.pop("config_hash", None)
        train = data.pop("train", {})
        if isinstance(train, dict):
            # JSON turns the epoch tuple into a list.
            if "save_epochs" in train:
                train["save_epochs"] = tuple(train["save_epochs"])
            train_cfg = TrainConfig(**train)
        else:
            train_cfg = train
        return cls(train=train_cfg, **data)


def code_fingerprint(root: str | Path = ".") -> str:
    """Stable hash of the algorithm sources and dependency declaration.

    The fingerprint intentionally excludes the git commit: freezing a manifest is
    itself a commit, so including ``HEAD`` would make every manifest stale as soon
    as it is committed. The commit and dirty state are recorded separately in each
    run's ``environment.json``.
    """
    root = Path(root)
    digest = hashlib.sha256()
    candidates = [root / "pyproject.toml", root / "uv.lock"]
    package = root / "src" / "ram_experiments"
    candidates.extend(sorted(path for path in package.rglob("*.py") if path.is_file()))
    found = False
    for path in candidates:
        if not path.is_file():
            continue
        found = True
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest() if found else "unknown"


def git_revision(root: str | Path = ".") -> dict[str, Any]:
    """Observed git commit and dirty state, recorded for provenance only."""
    root = Path(root)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def feasibility_tolerance(tau: float) -> float:
    """Primary float32 training tolerance from the specification."""
    return 1e-6 + 1e-5 * max(1.0, abs(tau))


def strict_tolerance(tau: float) -> float:
    """Float64 E0 tolerance from the specification."""
    return 1e-10 + 1e-10 * max(1.0, abs(tau))


def protocol_summary() -> dict[str, Any]:
    """Machine-readable snapshot used in run directories and reports."""
    return {
        "protocol": PROTOCOL_VERSION,
        "architectures": {name: list(dims) for name, dims in ARCHITECTURES.items()},
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "alphas": list(ALPHAS),
        "methods": list(METHODS),
        "wd_grid": list(WD_GRID),
        "nuc_grid": list(NUC_GRID),
        "train": TrainConfig().as_dict(),
        "split_seed": SPLIT_SEED,
    }
