"""Deterministic data preparation for MNIST and Fashion-MNIST.

The protocol fixes the split, the normalization and the pilot prefixes. This
module never touches the official test labels for model selection: test arrays
are returned separately and the runner only evaluates them when explicitly
allowed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ram_experiments.config import (
    MAIN_TRAIN_SIZE,
    MAIN_VAL_SIZE,
    PILOT_TRAIN_SIZE,
    PILOT_VAL_SIZE,
    SPLIT_SEED,
)

DATASET_ALIASES = {
    "mnist": "mnist",
    "fashion": "fashion_mnist",
    "fashion_mnist": "fashion_mnist",
    "fashion-mnist": "fashion_mnist",
}

N_CLASSES = 10


@dataclass
class DatasetBundle:
    """Flattened image tensors and labels for one protocol split."""

    name: str
    split: str
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    mean: float
    std: float
    indices: dict[str, list[int]] | None = None
    hashes: dict[str, str] | None = None

    @property
    def input_dim(self) -> int:
        return int(self.train_x.shape[1])

    def loaders(self, batch_size: int, seed: int, epoch: int = 0) -> tuple[DataLoader, DataLoader]:
        """Train loader with a precomputed permutation and a validation loader."""
        generator = torch.Generator()
        generator.manual_seed(int(seed) * 1_000_003 + int(epoch))
        permutation = torch.randperm(self.train_x.shape[0], generator=generator)
        train_dataset = TensorDataset(self.train_x, self.train_y)
        val_dataset = TensorDataset(self.val_x, self.val_y)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=permutation.tolist(),
            drop_last=False,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1024,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        return train_loader, val_loader

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "split": self.split,
            "train_size": int(self.train_x.shape[0]),
            "val_size": int(self.val_x.shape[0]),
            "test_size": int(self.test_x.shape[0]),
            "input_dim": self.input_dim,
            "mean": float(self.mean),
            "std": float(self.std),
            "hashes": self.hashes or {},
        }


def canonical_dataset_name(name: str) -> str:
    key = name.lower().strip()
    if key not in DATASET_ALIASES:
        raise KeyError(f"unknown dataset {name!r}; expected one of {sorted(DATASET_ALIASES)}")
    return DATASET_ALIASES[key]


def _load_torchvision(name: str, root: Path, download: bool, train: bool):
    from torchvision import datasets, transforms

    dataset_cls = datasets.MNIST if name == "mnist" else datasets.FashionMNIST
    return dataset_cls(
        root=str(root),
        train=train,
        download=download,
        transform=transforms.ToTensor(),
    )


def _to_arrays(dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    images = dataset.data.numpy().astype(np.uint8, copy=False)
    labels = np.asarray(dataset.targets, dtype=np.int64)
    return images.reshape(images.shape[0], -1), labels


def stratified_validation_counts(labels: np.ndarray, total: int) -> list[int]:
    """Per-class validation counts, proportional with remainder in class order."""
    _classes, counts = np.unique(labels, return_counts=True)
    n_total = int(labels.size)
    exact = [total * int(count) / n_total for count in counts]
    base = [int(np.floor(value)) for value in exact]
    remainder = total - sum(base)
    for index in range(remainder):
        base[index % len(base)] += 1
    return base


def stratified_split(
    labels: np.ndarray,
    total_validation: int = MAIN_VAL_SIZE,
    seed: int = SPLIT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic per-class split into train/validation indices."""
    classes = np.unique(labels)
    counts = stratified_validation_counts(labels, total_validation)
    rng = np.random.Generator(np.random.PCG64(seed))
    train_indices: list[int] = []
    val_indices: list[int] = []
    for class_label, n_val in zip(classes, counts, strict=True):
        class_indices = np.flatnonzero(labels == class_label)
        permutation = rng.permutation(class_indices)
        val_indices.extend(int(value) for value in permutation[:n_val])
        train_indices.extend(int(value) for value in permutation[n_val:])
    return np.sort(np.asarray(train_indices)), np.sort(np.asarray(val_indices))


def stratified_prefix(
    indices: np.ndarray,
    labels: np.ndarray,
    total: int,
) -> np.ndarray:
    """First ``total`` examples of a stratified subset, preserving split order."""
    selected: list[int] = []
    classes, counts = np.unique(labels[indices], return_counts=True)
    n_total = int(indices.size)
    exact = [total * int(count) / n_total for count in counts]
    per_class = [int(np.floor(value)) for value in exact]
    remainder = total - sum(per_class)
    for index in range(remainder):
        per_class[index % len(per_class)] += 1
    for class_label, n_class in zip(classes, per_class, strict=True):
        class_indices = indices[labels[indices] == class_label]
        selected.extend(int(value) for value in class_indices[:n_class])
    return np.sort(np.asarray(selected, dtype=np.int64))


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def build_bundles(
    name: str,
    images_train: np.ndarray,
    labels_train: np.ndarray,
    images_test: np.ndarray,
    labels_test: np.ndarray,
) -> dict[str, DatasetBundle]:
    """Build main and pilot bundles from the official training/test arrays."""
    name = canonical_dataset_name(name)
    main_train_idx, val_idx = stratified_split(labels_train, MAIN_VAL_SIZE, SPLIT_SEED)
    if main_train_idx.size != MAIN_TRAIN_SIZE:
        raise ValueError(f"expected {MAIN_TRAIN_SIZE} training examples, got {main_train_idx.size}")
    pilot_train_idx = stratified_prefix(main_train_idx, labels_train, PILOT_TRAIN_SIZE)
    pilot_val_idx = stratified_prefix(val_idx, labels_train, PILOT_VAL_SIZE)

    train_pixels = images_train[main_train_idx].astype(np.float32) / 255.0
    mean = float(train_pixels.mean())
    std = float(train_pixels.std())
    if std == 0.0:
        raise ValueError("degenerate normalization: zero standard deviation")

    def make(split: str, train_idx: np.ndarray, val_idx_local: np.ndarray) -> DatasetBundle:
        train_x = (images_train[train_idx].astype(np.float32) / 255.0 - mean) / std
        val_x = (images_train[val_idx_local].astype(np.float32) / 255.0 - mean) / std
        test_x = (images_test.astype(np.float32) / 255.0 - mean) / std
        return DatasetBundle(
            name=name,
            split=split,
            train_x=torch.from_numpy(train_x.astype(np.float32)),
            train_y=torch.from_numpy(labels_train[train_idx].astype(np.int64)),
            val_x=torch.from_numpy(val_x.astype(np.float32)),
            val_y=torch.from_numpy(labels_train[val_idx_local].astype(np.int64)),
            test_x=torch.from_numpy(test_x.astype(np.float32)),
            test_y=torch.from_numpy(labels_test.astype(np.int64)),
            mean=mean,
            std=std,
            indices={
                "train": [int(value) for value in train_idx],
                "val": [int(value) for value in val_idx_local],
            },
            hashes={
                "train": _sha256(images_train[train_idx]),
                "val": _sha256(images_train[val_idx_local]),
                "test": _sha256(images_test),
            },
        )

    return {
        "main": make("main", main_train_idx, val_idx),
        "pilot": make("pilot", pilot_train_idx, pilot_val_idx),
    }


def prepare_dataset(
    name: str,
    root: str | Path = "data/raw",
    out_dir: str | Path = "data/processed",
    download: bool = True,
) -> dict[str, Path]:
    """Download (optionally) and persist the protocol splits as ``.npz`` files."""
    canonical = canonical_dataset_name(name)
    root = Path(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = _load_torchvision(canonical, root, download=download, train=True)
    test = _load_torchvision(canonical, root, download=download, train=False)
    images_train, labels_train = _to_arrays(train)
    images_test, labels_test = _to_arrays(test)
    bundles = build_bundles(canonical, images_train, labels_train, images_test, labels_test)

    paths: dict[str, Path] = {}
    for split, bundle in bundles.items():
        path = out_dir / f"{canonical}_{split}.npz"
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
            indices_train=np.asarray(bundle.indices["train"]),
            indices_val=np.asarray(bundle.indices["val"]),
            hashes=json.dumps(bundle.hashes),
            name=canonical,
            split=split,
        )
        paths[split] = path
    manifest = {
        "dataset": canonical,
        "split_seed": SPLIT_SEED,
        "files": {split: str(path) for split, path in paths.items()},
        "bundles": {split: bundle.metadata() for split, bundle in bundles.items()},
    }
    (out_dir / f"{canonical}_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return paths


def load_bundle(
    name: str,
    split: str = "main",
    processed_dir: str | Path = "data/processed",
) -> DatasetBundle:
    """Load a previously prepared bundle."""
    canonical = canonical_dataset_name(name)
    path = Path(processed_dir) / f"{canonical}_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run "
            f"`python -m ram_experiments.cli prepare-data --datasets {canonical}`"
        )
    with np.load(path, allow_pickle=False) as payload:
        return DatasetBundle(
            name=canonical,
            split=split,
            train_x=torch.from_numpy(payload["train_x"].astype(np.float32)),
            train_y=torch.from_numpy(payload["train_y"].astype(np.int64)),
            val_x=torch.from_numpy(payload["val_x"].astype(np.float32)),
            val_y=torch.from_numpy(payload["val_y"].astype(np.int64)),
            test_x=torch.from_numpy(payload["test_x"].astype(np.float32)),
            test_y=torch.from_numpy(payload["test_y"].astype(np.int64)),
            mean=float(payload["mean"]),
            std=float(payload["std"]),
            indices={
                "train": payload["indices_train"].astype(int).tolist(),
                "val": payload["indices_val"].astype(int).tolist(),
            },
            hashes=json.loads(str(payload["hashes"])),
        )


def synthetic_bundle(
    n_train: int = 256,
    n_val: int = 64,
    n_test: int = 64,
    input_dim: int = 784,
    n_classes: int = 10,
    seed: int = 0,
) -> DatasetBundle:
    """Small deterministic bundle for tests; not a benchmark dataset."""
    generator = torch.Generator().manual_seed(seed)
    train_x = torch.randn(n_train, input_dim, generator=generator)
    val_x = torch.randn(n_val, input_dim, generator=generator)
    test_x = torch.randn(n_test, input_dim, generator=generator)
    train_y = torch.randint(0, n_classes, (n_train,), generator=generator)
    val_y = torch.randint(0, n_classes, (n_val,), generator=generator)
    test_y = torch.randint(0, n_classes, (n_test,), generator=generator)
    return DatasetBundle(
        name="synthetic",
        split="synthetic",
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        test_x=test_x,
        test_y=test_y,
        mean=0.0,
        std=1.0,
        indices={"train": list(range(n_train)), "val": list(range(n_val))},
        hashes={},
    )
