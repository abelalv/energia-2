"""Training loop, ADMM/PSGD-M schedules, checkpoints and resumption."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ram_experiments.config import RunConfig, git_revision
from ram_experiments.data import DatasetBundle
from ram_experiments.energy import omega
from ram_experiments.evaluate import evaluation_record, spectral_report
from ram_experiments.methods import (
    BudgetPhase,
    adapt_rho,
    augmented_penalty,
    budget_phase,
    data_loss,
    initial_tau0,
    method_metadata,
    objective_term,
    optimizer_for,
    residual_norms,
)
from ram_experiments.models import (
    MLP,
    apply_spectral_norm,
    effective_clone,
    effective_state_dict,
    make_model,
    parameters_fingerprint,
)
from ram_experiments.projection import (
    ProjectionPolicy,
    allocate_radii,
    project_joint,
    project_joint_torch,
    project_layerwise,
)


class NumericalFailure(RuntimeError):
    """Raised when a non-finite value or a projection failure is detected."""


@dataclass
class TrainResult:
    status: str
    run_dir: Path
    epochs_completed: int
    metrics: list[dict[str, Any]]
    message: str = ""


def set_determinism(seed: int, tf32: bool = False) -> None:
    """Fix Python/NumPy/Torch seeds and disable nondeterministic CUDA paths."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = bool(tf32)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto``/``cpu``/``cuda``/``mps`` to an available device."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def environment_record(device: torch.device) -> dict[str, Any]:
    """Observed environment, not a compatibility guarantee."""
    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        }
    return {
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version()
        if hasattr(torch.backends, "cudnn")
        else None,
        "device": str(device),
        "gpu": gpu,
        "tf32": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "git": git_revision(),
    }


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _json_dump(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    os.replace(temporary, path)


def _project_weights_in_place(
    model: MLP,
    tau: float,
    policy: ProjectionPolicy,
) -> dict[str, Any]:
    weights = [layer.weight for layer in model.layers]
    projected, record = project_joint_torch(weights, tau, policy=policy)
    with torch.no_grad():
        for layer, new_weight in zip(model.layers, projected, strict=True):
            layer.weight.copy_(new_weight)
    return record


def _project_admm_z(
    weights: list[torch.Tensor],
    u: list[torch.Tensor],
    tau: float,
    policy: ProjectionPolicy,
    allocation: str | None,
    reference: list[torch.Tensor] | None,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    shifted = [weight.detach() + dual for weight, dual in zip(weights, u, strict=True)]
    if allocation is None:
        return project_joint_torch(shifted, tau, policy=policy)
    if reference is None:
        raise ValueError("layerwise allocation requires a reference weight list")
    radii = allocate_radii(
        [layer.detach().cpu().numpy() for layer in reference],
        tau,
        mode=allocation,
    )
    projected_np, record = project_layerwise(
        [layer.detach().cpu().numpy() for layer in shifted], radii
    )
    projected = [
        torch.as_tensor(array, dtype=weight.dtype, device=weight.device)
        for array, weight in zip(projected_np, shifted, strict=True)
    ]
    record["allocation"] = allocation
    return projected, record


def _check_finite(value: float, label: str) -> None:
    if not math.isfinite(value):
        raise NumericalFailure(f"non-finite {label}: {value!r}")


def _evaluate_split(
    model: MLP,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    from ram_experiments.evaluate import evaluate_model

    return evaluate_model(model, inputs, targets, device=device)


def train_run(
    bundle: DatasetBundle,
    config: RunConfig,
    run_dir: str | Path,
    *,
    device: str | torch.device = "auto",
    tau_final: float | None = None,
    tau_reference: float | None = None,
    resume: bool = False,
    max_seconds: float | None = None,
    allow_test: bool = False,
    init_state: dict[str, torch.Tensor] | None = None,
    reference_weights: list[torch.Tensor] | None = None,
) -> TrainResult:
    """Run one configuration to completion or until the session time budget.

    ``tau_final`` is required for ADMM and PSGD-M; it is normally
    ``alpha * Omega(W_STD_reference)``. ``tau_reference`` is only recorded.
    """
    if config.uses_budget and tau_final is None:
        raise ValueError(f"{config.method} requires tau_final")

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch_device = resolve_device(str(device))
    set_determinism(config.seed, tf32=config.train.tf32)
    started = time.monotonic()

    model = make_model(config.architecture, config.seed)
    if init_state is not None:
        model.load_state_dict(init_state)
    if config.method == "SN":
        apply_spectral_norm(
            model,
            n_power_iterations=config.train.sn_n_power_iterations,
            eps=config.train.sn_eps,
        )
    model.to(torch_device)
    optimizer = optimizer_for(model, config.train)
    policy = ProjectionPolicy(
        device=config.train.projection_device,
        dtype=config.train.projection_dtype,
    )

    z: list[torch.Tensor] | None = None
    u: list[torch.Tensor] | None = None
    rho = float(config.train.admm_rho)
    tau0: float | None = None
    start_epoch = 0
    metrics: list[dict[str, Any]] = []

    checkpoint_path = run_dir / "checkpoint.pt"
    if resume and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"])
        z = [tensor.to(torch_device) for tensor in payload["z"]] if payload.get("z") else None
        u = [tensor.to(torch_device) for tensor in payload["u"]] if payload.get("u") else None
        rho = float(payload.get("rho", rho))
        tau0 = payload.get("tau0")
        if payload.get("rng"):
            random.setstate(payload["rng"]["python"])
            np.random.set_state(payload["rng"]["numpy"])
            torch.set_rng_state(payload["rng"]["torch"])
            if torch.cuda.is_available() and payload["rng"].get("cuda") is not None:
                torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
        if (run_dir / "epochs.jsonl").exists():
            metrics = [
                json.loads(line)
                for line in (run_dir / "epochs.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    config_payload = config.to_dict()
    config_payload.update(
        {
            "tau_final": tau_final,
            "tau_reference": tau_reference,
            "data": bundle.metadata(),
            "environment": environment_record(torch_device),
            "method_metadata": method_metadata(config.method),
            "started_epoch": start_epoch,
        }
    )
    _json_dump(config_payload, run_dir / "config.json")

    def save_checkpoint(epoch: int, complete: bool) -> None:
        payload = {
            "epoch": epoch,
            "complete": complete,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "z": [tensor.detach().cpu() for tensor in z] if z is not None else None,
            "u": [tensor.detach().cpu() for tensor in u] if u is not None else None,
            "rho": rho,
            "tau0": tau0,
            "tau_final": tau_final,
            "config_hash": config.config_hash(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        _atomic_torch_save(payload, checkpoint_path)

    if start_epoch == 0:
        save_checkpoint(0, complete=False)

    status = "completed"
    message = ""
    training_config = config.train

    try:
        for epoch in range(start_epoch + 1, training_config.epochs + 1):
            if max_seconds is not None and time.monotonic() - started > max_seconds:
                status = "incomplete_time_budget"
                message = f"stopped before epoch {epoch}"
                break

            if config.uses_budget:
                phase = budget_phase(
                    epoch,
                    tau0 if epoch > training_config.warmup_epochs else None,
                    tau_final,
                    training_config,
                )
            else:
                phase = BudgetPhase(epoch=epoch, phase="train", tau=None)
            rho_used = rho
            z_old = [tensor.detach().clone() for tensor in z] if z is not None else None
            u_old = [tensor.detach().clone() for tensor in u] if u is not None else None
            active = config.uses_budget and phase.tau is not None
            projection_count = 0

            if config.method == "PSGD-M" and active:
                record = _project_weights_in_place(model, float(phase.tau), policy)
                projection_count += 1
                if not record["feasible"]:
                    raise NumericalFailure(f"PSGD-M entry projection infeasible: {record}")

            train_loader, _ = bundle.loaders(training_config.batch_size, config.seed, epoch)
            model.train()
            running_loss = 0.0
            running_data = 0.0
            running_reg = 0.0
            running_aug = 0.0
            running_correct = 0
            running_total = 0
            for inputs, targets in train_loader:
                inputs = inputs.to(torch_device)
                targets = targets.to(torch_device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                data_term = data_loss(logits, targets)
                loss = data_term
                weights = [layer.weight for layer in model.layers]
                regularization = objective_term(config.method, model, weights, config.reg_lambda)
                if regularization is not None:
                    loss = loss + regularization
                augmented = None
                if config.method == "ADMM" and active:
                    augmented = augmented_penalty(weights, z, u, rho_used)  # type: ignore[arg-type]
                    loss = loss + augmented
                _check_finite(float(loss.detach().cpu()), "training loss")
                loss.backward()
                optimizer.step()
                if config.method == "PSGD-M" and active:
                    record = _project_weights_in_place(model, float(phase.tau), policy)
                    projection_count += 1
                    if not record["feasible"]:
                        raise NumericalFailure(f"PSGD-M minibatch projection infeasible: {record}")
                batch_size = int(targets.size(0))
                running_loss += float(loss.detach().cpu()) * batch_size
                running_data += float(data_term.detach().cpu()) * batch_size
                running_reg += (
                    float(regularization.detach().cpu()) * batch_size
                    if regularization is not None
                    else 0.0
                )
                running_aug += (
                    float(augmented.detach().cpu()) * batch_size if augmented is not None else 0.0
                )
                running_correct += int((logits.argmax(dim=1) == targets).sum().cpu())
                running_total += int(targets.size(0))

            train_loss = running_loss / max(running_total, 1)
            train_data_loss = running_data / max(running_total, 1)
            train_reg_term = running_reg / max(running_total, 1)
            train_aug_term = running_aug / max(running_total, 1)
            train_accuracy = running_correct / max(running_total, 1)
            weights_now = [layer.weight.detach().clone() for layer in model.layers]
            omega_w = omega([tensor.cpu().numpy() for tensor in weights_now])
            _check_finite(omega_w, "Omega(W)")
            record: dict[str, Any] = {
                "epoch": epoch,
                "phase": phase.phase,
                "tau": phase.tau,
                "tau0": tau0,
                "tau_final": tau_final,
                "objective_id": method_metadata(config.method)["objective_id"],
                "train_loss": train_loss,
                "train_data_loss": train_data_loss,
                "train_reg_term": train_reg_term,
                "train_aug_term": train_aug_term,
                "train_accuracy": train_accuracy,
                "omega_W": omega_w,
                "violation_W": (
                    None
                    if tau_final is None
                    else max(omega_w - (phase.tau if phase.tau is not None else tau_final), 0.0)
                ),
                "rho_used": rho_used,
                "rho_next": rho,
                "projection_count": projection_count,
                "elapsed_seconds": time.monotonic() - started,
            }

            if config.method in {"ADMM", "PSGD-M"} and epoch == training_config.warmup_epochs:
                tau0 = initial_tau0(omega_w, float(tau_final))
                record.update({"tau0": tau0})
                if config.method == "ADMM":
                    z = [weight.clone() for weight in weights_now]
                    u = [torch.zeros_like(weight) for weight in weights_now]
                    rho = float(training_config.admm_rho)
                    record.update({"rho_next": rho, "z_initialized": True})

            if config.method == "ADMM" and active:
                assert z is not None and u is not None and z_old is not None and u_old is not None
                z_new, projection_record = _project_admm_z(
                    [layer.weight for layer in model.layers],
                    u_old,
                    float(phase.tau),
                    policy,
                    config.allocation,
                    reference_weights,
                )
                primal, dual = residual_norms(
                    [layer.weight.detach() for layer in model.layers], z_new, z_old, rho_used
                )
                u_new = [
                    old + (weight.detach() - z_layer)
                    for old, weight, z_layer in zip(
                        u_old, [layer.weight for layer in model.layers], z_new, strict=True
                    )
                ]
                rho_next, rescale = adapt_rho(
                    primal,
                    dual,
                    rho_used,
                    balance=training_config.rho_balance,
                    factor=training_config.rho_factor,
                    rho_min=training_config.rho_min,
                    rho_max=training_config.rho_max,
                )
                u = [tensor * rescale for tensor in u_new]
                z = z_new
                rho = rho_next
                record.update(
                    {
                        "rho_used": rho_used,
                        "rho_next": rho,
                        "dual_rescale_factor": rescale,
                        "primal_residual": primal,
                        "dual_residual": dual,
                        "projection_threshold": projection_record.get("threshold_t"),
                        "omega_Z": projection_record.get("omega_after"),
                        "z_feasible": projection_record.get("feasible"),
                        "projection_count": projection_count + 1,
                        "allocation": config.allocation,
                    }
                )

            val = _evaluate_split(model, bundle.val_x, bundle.val_y, torch_device)
            record.update({"val_loss": val["loss"], "val_accuracy": val["accuracy"]})
            metrics.append(record)
            with (run_dir / "epochs.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")

            if epoch in training_config.save_epochs or epoch == training_config.epochs:
                spectra = spectral_report(weights_now)
                np.savez_compressed(
                    run_dir / f"spectra_epoch_{epoch:03d}.npz",
                    **{
                        f"layer_{index}_spectrum": np.asarray(layer["spectrum"])
                        for index, layer in enumerate(spectra["layers"])
                    },
                )
                _json_dump(spectra, run_dir / f"spectra_epoch_{epoch:03d}.json")

            save_checkpoint(epoch, complete=epoch == training_config.epochs)

    except NumericalFailure as error:
        status = "numerical_failure"
        message = str(error)
        save_checkpoint(start_epoch if not metrics else int(metrics[-1]["epoch"]), complete=False)
    except torch.cuda.OutOfMemoryError as error:
        status = "resource_failure"
        message = str(error)
        save_checkpoint(start_epoch if not metrics else int(metrics[-1]["epoch"]), complete=False)

    epochs_completed = int(metrics[-1]["epoch"]) if metrics else start_epoch
    if (
        max_seconds is not None
        and status == "completed"
        and time.monotonic() - started > max_seconds
    ):
        message = "completed after exceeding the soft session budget"

    # Final artifacts only when the scheduled horizon is complete.
    if status == "completed" and epochs_completed >= training_config.epochs:
        _write_final_artifacts(
            model=model,
            config=config,
            bundle=bundle,
            run_dir=run_dir,
            tau_final=tau_final,
            z=z,
            device=torch_device,
            policy=policy,
            allow_test=allow_test,
        )

    status_payload = {
        "status": status,
        "message": message,
        "epochs_completed": epochs_completed,
        "config_hash": config.config_hash(),
        "run_id": config.run_id(),
        "parameters_fingerprint": parameters_fingerprint(model),
        "elapsed_seconds": time.monotonic() - started,
    }
    _json_dump(status_payload, run_dir / "status.json")
    _write_metrics_csv(metrics, run_dir / "metrics.csv")
    return TrainResult(
        status=status,
        run_dir=run_dir,
        epochs_completed=epochs_completed,
        metrics=metrics,
        message=message,
    )


def _write_metrics_csv(metrics: list[dict[str, Any]], path: Path) -> None:
    if not metrics:
        return
    fields = [
        "epoch",
        "phase",
        "tau",
        "objective_id",
        "train_loss",
        "train_data_loss",
        "train_reg_term",
        "train_aug_term",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "omega_W",
        "violation_W",
        "rho_used",
        "rho_next",
        "projection_count",
        "elapsed_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)


def _write_final_artifacts(
    *,
    model: MLP,
    config: RunConfig,
    bundle: DatasetBundle,
    run_dir: Path,
    tau_final: float | None,
    z: list[torch.Tensor] | None,
    device: torch.device,
    policy: ProjectionPolicy,
    allow_test: bool,
) -> None:
    """Evaluate the three prespecified variants and store their state."""
    base = effective_clone(model).to(device)
    variants: dict[str, MLP] = {"W": base}

    if z is not None:
        z_model = MLP(base.dims)
        state = effective_state_dict(base)
        for index, _layer in enumerate(z_model.layers):
            state[f"layers.{index}.weight"] = z[index].detach().clone()
        z_model.load_state_dict(state)
        variants["Z"] = z_model.to(device)

    if tau_final is not None:
        weights = [layer.weight.detach() for layer in base.layers]
        projected_np, record = project_joint(
            [tensor.cpu().numpy() for tensor in weights], tau_final, policy=policy
        )
        projected_model = MLP(base.dims)
        state = effective_state_dict(base)
        for index, array in enumerate(projected_np):
            state[f"layers.{index}.weight"] = torch.as_tensor(
                array, dtype=state[f"layers.{index}.weight"].dtype
            )
        projected_model.load_state_dict(state)
        variants["PW"] = projected_model.to(device)
        _json_dump(record, run_dir / "final_projection.json")

    records = {}
    for name, variant_model in variants.items():
        records[name] = evaluation_record(
            variant=name,
            model=variant_model,
            bundle=bundle,
            tau_final=tau_final,
            device=device,
            allow_test=allow_test,
        )
    _json_dump(records, run_dir / "final_evaluation.json")

    payload = {
        "W": effective_state_dict(base),
        **{
            name: effective_state_dict(variant) for name, variant in variants.items() if name != "W"
        },
        "z": [tensor.detach().cpu() for tensor in z] if z is not None else None,
        "tau_final": tau_final,
        "config": config.to_dict(),
    }
    _atomic_torch_save(payload, run_dir / "final.pt")
