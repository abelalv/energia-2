"""Manifest execution with reference resolution, resume and session budgets."""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ram_experiments.config import RunConfig
from ram_experiments.data import load_bundle
from ram_experiments.energy import omega
from ram_experiments.manifest import read_manifest, shard_manifest, validate_manifest
from ram_experiments.models import make_model
from ram_experiments.train import environment_record, resolve_device, train_run


@dataclass
class RunOutcome:
    run_id: str
    stage: str
    method: str
    dataset: str
    architecture: str
    seed: int
    status: str
    seconds: float
    run_dir: str
    message: str = ""


def _status_path(run_dir: Path) -> Path:
    return run_dir / "status.json"


def _is_complete(run_dir: Path) -> bool:
    status_path = _status_path(run_dir)
    if not status_path.exists():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return status.get("status") == "completed" and (run_dir / "final.pt").exists()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _init_state_path(run_root: Path, run: RunConfig) -> Path:
    return run_root / "init" / f"{run.dataset}-{run.architecture}-s{run.seed}.pt"


def _ensure_init_state(run_root: Path, run: RunConfig) -> dict[str, torch.Tensor]:
    path = _init_state_path(run_root, run)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    model = make_model(run.architecture, run.seed)
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save(state, path)
    return state


def _find_std_reference(run: RunConfig, runs: list[RunConfig]) -> RunConfig | None:
    target_stage = "pilot" if run.stage == "pilot" else "E2"
    candidates = [
        candidate
        for candidate in runs
        if candidate.stage == target_stage
        and candidate.method == "STD"
        and candidate.dataset == run.dataset
        and candidate.architecture == run.architecture
        and candidate.seed == run.seed
    ]
    return candidates[0] if candidates else None


def _std_omega(run_dir: Path) -> float:
    final_eval = run_dir / "final_evaluation.json"
    if final_eval.exists():
        payload = _read_json(final_eval)
        if "W" in payload:
            return float(payload["W"]["spectral"]["omega"])
    final_pt = run_dir / "final.pt"
    if final_pt.exists():
        payload = torch.load(final_pt, map_location="cpu", weights_only=False)
        state = payload["W"]
        weights = [value for key, value in state.items() if key.endswith(".weight")]
        return omega([weight.numpy() for weight in weights])
    raise FileNotFoundError(f"no STD artifact in {run_dir}")


def _load_std_weights(run_dir: Path, device: torch.device) -> list[torch.Tensor]:
    payload = torch.load(run_dir / "final.pt", map_location="cpu", weights_only=False)
    state = payload["W"]
    weight_values = [value for key, value in state.items() if key.endswith(".weight")]
    return [value.to(device) for value in weight_values]


def _prepare_continuation(run: RunConfig, runs: list[RunConfig], run_root: Path) -> None:
    if run.parent_run_id is None:
        raise ValueError("a continuation requires parent_run_id")
    parents = [candidate for candidate in runs if candidate.run_id() == run.parent_run_id]
    if not parents:
        raise ValueError(f"parent {run.parent_run_id} is not in the manifest")
    parent_dir = run_root / parents[0].stage / parents[0].run_id()
    parent_checkpoint = parent_dir / "checkpoint.pt"
    if not parent_checkpoint.exists():
        raise FileNotFoundError(
            f"parent checkpoint missing: {parent_checkpoint}; run stage E2 first"
        )
    target_dir = run_root / run.stage / run.run_id()
    target_dir.mkdir(parents=True, exist_ok=True)
    target_checkpoint = target_dir / "checkpoint.pt"
    if not target_checkpoint.exists():
        target_checkpoint.write_bytes(parent_checkpoint.read_bytes())


def _run_single(
    run: RunConfig,
    runs: list[RunConfig],
    *,
    run_root: Path,
    data_dir: Path,
    split: str,
    device: str,
    resume: bool,
    remaining_seconds: float | None,
    allow_test: bool,
    fingerprint_mismatch: bool,
) -> RunOutcome:
    started = time.monotonic()
    run_dir = run_root / run.stage / run.run_id()
    run_dir.mkdir(parents=True, exist_ok=True)
    if fingerprint_mismatch:
        raise RuntimeError(
            "manifest code fingerprint does not match the current sources; "
            "regenerate the manifest or check out the recorded commit"
        )

    if _is_complete(run_dir):
        return RunOutcome(
            run_id=run.run_id(),
            stage=run.stage,
            method=run.method,
            dataset=run.dataset,
            architecture=run.architecture,
            seed=run.seed,
            status="reused",
            seconds=0.0,
            run_dir=str(run_dir),
        )

    bundle = load_bundle(run.dataset, split=split, processed_dir=data_dir)
    torch_device = resolve_device(device)
    tau_final: float | None = None
    reference_weights: list[torch.Tensor] | None = None

    if run.method != "STD" and (run.uses_budget or run.tau_source is not None):
        if run.tau_source == "parent":
            _prepare_continuation(run, runs, run_root)
            parent_dir = run_root / "E2" / str(run.parent_run_id)
            parent_config = _read_json(parent_dir / "config.json")
            tau_final = float(parent_config["tau_final"])
        else:
            reference_spec = _find_std_reference(run, runs)
            if reference_spec is None:
                raise ValueError(f"no STD reference found for {run.run_id()}")
            reference_dir = run_root / reference_spec.stage / reference_spec.run_id()
            if not _is_complete(reference_dir):
                reference_outcome = _run_single(
                    reference_spec,
                    runs,
                    run_root=run_root,
                    data_dir=data_dir,
                    split=split if reference_spec.stage == "pilot" else "main",
                    device=device,
                    resume=resume,
                    remaining_seconds=remaining_seconds,
                    allow_test=False,
                    fingerprint_mismatch=fingerprint_mismatch,
                )
                if reference_outcome.status not in {"completed", "reused"}:
                    raise RuntimeError(
                        f"STD reference {reference_spec.run_id()} ended with "
                        f"{reference_outcome.status}"
                    )
            scale = float(run.alpha) if run.alpha is not None else 1.0
            tau_final = scale * _std_omega(reference_dir)
            if run.allocation in {"weights", "reference"}:
                reference_weights = _load_std_weights(reference_dir, torch_device)

    if run.uses_budget and tau_final is None:
        raise ValueError(f"{run.method} requires a budget")

    init_state = _ensure_init_state(run_root, run)
    result = train_run(
        bundle,
        run,
        run_dir,
        device=device,
        tau_final=tau_final,
        resume=resume,
        max_seconds=remaining_seconds,
        allow_test=allow_test,
        init_state=init_state,
        reference_weights=reference_weights,
    )
    return RunOutcome(
        run_id=run.run_id(),
        stage=run.stage,
        method=run.method,
        dataset=run.dataset,
        architecture=run.architecture,
        seed=run.seed,
        status=result.status,
        seconds=time.monotonic() - started,
        run_dir=str(run_dir),
        message=result.message,
    )


def execute_manifest(
    manifest_path: str | Path,
    *,
    stages: set[str] | None = None,
    run_root: str | Path = "runs",
    data_dir: str | Path = "data/processed",
    device: str = "auto",
    shard: int = 0,
    total_shards: int = 1,
    max_seconds: float | None = None,
    resume: bool = True,
    allow_test: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
    skip_fingerprint_check: bool = False,
) -> dict[str, Any]:
    """Execute a manifest in a bounded session.

    Returns a session summary and writes it next to the run directories.
    """
    runs = read_manifest(manifest_path)
    all_runs = list(runs)
    if stages:
        runs = [run for run in runs if run.stage in stages]
    inventory = validate_manifest(runs)
    if not inventory["valid"]:
        raise ValueError(f"manifest has duplicate configurations: {inventory}")
    if total_shards > 1:
        runs = shard_manifest(runs, shard, total_shards)
    if limit is not None:
        runs = runs[:limit]

    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    session_started = time.monotonic()
    outcomes: list[RunOutcome] = []
    planned = [run.run_id() for run in runs]

    if dry_run:
        return {
            "dry_run": True,
            "n_planned": len(planned),
            "planned": planned,
            "inventory": inventory,
        }

    for run in runs:
        elapsed = time.monotonic() - session_started
        if max_seconds is not None and elapsed >= max_seconds:
            outcomes.append(
                RunOutcome(
                    run_id=run.run_id(),
                    stage=run.stage,
                    method=run.method,
                    dataset=run.dataset,
                    architecture=run.architecture,
                    seed=run.seed,
                    status="not_started_time_budget",
                    seconds=0.0,
                    run_dir=str(run_root / run.stage / run.run_id()),
                )
            )
            continue
        remaining = None if max_seconds is None else max_seconds - elapsed
        try:
            outcome = _run_single(
                run,
                all_runs,
                run_root=run_root,
                data_dir=Path(data_dir),
                split="pilot" if run.stage == "pilot" else "main",
                device=device,
                resume=resume,
                remaining_seconds=remaining,
                allow_test=allow_test,
                fingerprint_mismatch=False,
            )
        except Exception as error:
            outcome = RunOutcome(
                run_id=run.run_id(),
                stage=run.stage,
                method=run.method,
                dataset=run.dataset,
                architecture=run.architecture,
                seed=run.seed,
                status="error",
                seconds=time.monotonic() - session_started - elapsed,
                run_dir=str(run_root / run.stage / run.run_id()),
                message=f"{type(error).__name__}: {error}\n{traceback.format_exc()}",
            )
            run_dir = run_root / run.stage / run.run_id()
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "error.txt").write_text(outcome.message, encoding="utf-8")
        outcomes.append(outcome)
        if outcome.status == "incomplete_time_budget":
            break
        if outcome.status == "resource_failure":
            break

    summary = {
        "manifest": str(manifest_path),
        "stages": sorted(stages) if stages else None,
        "shard": shard,
        "total_shards": total_shards,
        "session_seconds": time.monotonic() - session_started,
        "n_planned": len(planned),
        "n_processed": len(outcomes),
        "counts": _count_statuses(outcomes),
        "environment": environment_record(resolve_device(device)),
        "runs": [asdict(outcome) for outcome in outcomes],
    }
    summary_path = run_root / f"session_{int(time.time())}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def _count_statuses(outcomes: list[RunOutcome]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for outcome in outcomes:
        counts[outcome.status] = counts.get(outcome.status, 0) + 1
    return counts
