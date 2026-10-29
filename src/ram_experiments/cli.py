"""Command-line interface for protocol RAM-2026-10-13-v1."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import torch

from ram_experiments.config import (
    ALPHAS,
    DATASETS,
    NUC_GRID,
    PROTOCOL_VERSION,
    SEEDS,
    WD_GRID,
    RunConfig,
    code_fingerprint,
    protocol_summary,
)
from ram_experiments.data import load_bundle, prepare_dataset
from ram_experiments.energy import omega, spectral_summary
from ram_experiments.evaluate import evaluate_model, feasibility_report
from ram_experiments.manifest import (
    expected_counts,
    generate_main_manifest,
    generate_pilot_manifest,
    read_manifest,
    validate_manifest,
    write_manifest,
)
from ram_experiments.models import MLP
from ram_experiments.runner import execute_manifest


def _parse_stages(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _cmd_manifest(args: argparse.Namespace) -> int:
    if args.stage == "pilot":
        runs = generate_pilot_manifest()
    elif args.stage == "main":
        runs = generate_main_manifest()
    else:
        raise ValueError("stage must be 'pilot' or 'main'")
    path = write_manifest(runs, args.out)
    report = validate_manifest(runs)
    print(json.dumps({"manifest": str(path), **report}, indent=2, sort_keys=True))
    return 0


def _cmd_inventory(args: argparse.Namespace) -> int:
    runs = read_manifest(args.manifest)
    report = validate_manifest(runs)
    report["expected"] = expected_counts()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


def _cmd_run(args: argparse.Namespace) -> int:
    summary = execute_manifest(
        args.manifest,
        stages=_parse_stages(args.stage),
        run_root=args.run_root,
        data_dir=args.data_dir,
        device=args.device,
        shard=args.shard,
        total_shards=args.total_shards,
        max_seconds=args.max_seconds,
        resume=args.resume,
        allow_test=args.allow_test,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _cmd_prepare_data(args: argparse.Namespace) -> int:
    outputs: dict[str, Any] = {}
    for name in args.datasets.split(","):
        name = name.strip()
        paths = prepare_dataset(
            name,
            root=args.raw_dir,
            out_dir=args.processed_dir,
            download=not args.no_download,
        )
        outputs[name] = {key: str(value) for key, value in paths.items()}
    print(json.dumps({"processed": outputs}, indent=2, sort_keys=True))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    suites = (args.suite or "E0,E1").split(",")
    selected: list[str] = []
    for suite in suites:
        mapping = {"E0": "tests/test_e0.py", "E1": "tests/test_e1.py"}
        path = mapping.get(suite.strip().upper())
        if path is None:
            raise ValueError(f"unknown suite {suite!r}")
        selected.append(path)
    command = [sys.executable, "-m", "pytest", *selected, "-q"]
    completed = subprocess.run(command, cwd=args.root, text=True, capture_output=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "suite": suites,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "status": "passed" if completed.returncode == 0 else "failed",
    }
    (out_dir / "E0_E1_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return completed.returncode


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _cmd_analyze(args: argparse.Namespace) -> int:
    runs = read_manifest(args.manifest)
    rows: list[dict[str, Any]] = []
    for run in runs:
        run_dir = Path(args.run_root) / run.stage / run.run_id()
        evaluation = _read_json(run_dir / "final_evaluation.json")
        if not evaluation:
            continue
        row: dict[str, Any] = {
            "run_id": run.run_id(),
            "stage": run.stage,
            "dataset": run.dataset,
            "architecture": run.architecture,
            "seed": run.seed,
            "method": run.method,
            "alpha": run.alpha,
            "reg_lambda": run.reg_lambda,
            "allocation": run.allocation,
            "epochs": run.train.epochs,
        }
        for variant in ("W", "Z", "PW"):
            payload = evaluation.get(variant)
            if not payload:
                continue
            prefix = variant.lower()
            row[f"{prefix}_val_loss"] = payload["validation"]["loss"]
            row[f"{prefix}_val_accuracy"] = payload["validation"]["accuracy"]
            row[f"{prefix}_omega"] = payload["spectral"]["omega"]
            feasibility = payload.get("feasibility", {})
            row[f"{prefix}_violation"] = feasibility.get("violation")
            row[f"{prefix}_feasible"] = feasibility.get("feasible")
        rows.append(row)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with (out_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    selection = _select_calibration(runs, Path(args.run_root))
    (out_dir / "calibration_selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"rows": len(rows), "selection": selection}, indent=2, sort_keys=True))
    return 0


def _select_calibration(runs, run_root: Path) -> dict[str, Any]:
    """Select one WD/NUC coefficient per dataset/architecture/alpha.

    Implements the protocol rule: minimize the mean over the five seeds of
    ``|Omega/tau - 1|``; ties go to lower mean validation loss, then to the smaller
    coefficient. A candidate is eligible only if all five seeds finished.
    """
    std_omega: dict[tuple[str, str, int], float] = {}
    records: dict[tuple[str, str, str, float, int], dict[str, float]] = {}
    for run in runs:
        run_dir = run_root / run.stage / run.run_id()
        evaluation = _read_json(run_dir / "final_evaluation.json")
        if not evaluation or "W" not in evaluation:
            continue
        tau_key = (run.dataset, run.architecture, run.seed)
        if run.method == "STD":
            std_omega[tau_key] = float(evaluation["W"]["spectral"]["omega"])
        if run.method in {"WD", "NUC"} and run.reg_lambda is not None:
            records[(run.dataset, run.architecture, run.method, run.reg_lambda, run.seed)] = {
                "omega": float(evaluation["W"]["spectral"]["omega"]),
                "val_loss": float(evaluation["W"]["validation"]["loss"]),
            }

    selection: dict[str, Any] = {}
    for dataset in DATASETS:
        for architecture in sorted({run.architecture for run in runs}):
            for method, grid in (("WD", WD_GRID), ("NUC", NUC_GRID)):
                for alpha in ALPHAS:
                    best: tuple[float, float, float] | None = None
                    for coefficient in grid:
                        distances: list[float] = []
                        losses: list[float] = []
                        for seed in SEEDS:
                            record = records.get((dataset, architecture, method, coefficient, seed))
                            key = (dataset, architecture, seed)
                            if record is None or key not in std_omega:
                                continue
                            tau = alpha * std_omega[key]
                            if tau <= 0.0:
                                continue
                            distances.append(abs(record["omega"] / tau - 1.0))
                            losses.append(record["val_loss"])
                        if len(distances) != len(SEEDS):
                            continue
                        mean_distance = sum(distances) / len(distances)
                        mean_loss = sum(losses) / len(losses)
                        candidate = (mean_distance, mean_loss, coefficient)
                        if best is None or candidate < best:
                            best = candidate
                    label = f"{dataset}/{architecture}/{method}/alpha={alpha}"
                    if best is None:
                        selection[label] = {"status": "no_energy_match", "n_seeds": 0}
                    else:
                        selection[label] = {
                            "coefficient": best[2],
                            "mean_energy_distance": best[0],
                            "mean_val_loss": best[1],
                            "n_seeds": len(SEEDS),
                        }
    return {
        "rule": "min mean |Omega/tau-1|; tie -> lower mean val loss; tie -> smaller coefficient",
        "selection": selection,
    }


def _cmd_export(args: argparse.Namespace) -> int:
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(Path(args.run_root).rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(Path(args.run_root).parent))
        if Path(args.manifest).exists():
            archive.write(args.manifest, Path(args.manifest).name)
    print(json.dumps({"archive": str(output), "bytes": output.stat().st_size}, indent=2))
    return 0


def _cmd_kaggle_plan(args: argparse.Namespace) -> int:
    runs = read_manifest(args.manifest)
    total_epochs = sum(run.train.epochs for run in runs)
    seconds_per_epoch = args.seconds_per_epoch
    estimate = total_epochs * seconds_per_epoch
    sessions = max(1, int(-(-estimate // args.session_seconds)))
    plan = {
        "n_runs": len(runs),
        "total_epochs": total_epochs,
        "seconds_per_epoch": seconds_per_epoch,
        "estimated_hours": estimate / 3600,
        "session_seconds": args.session_seconds,
        "sessions": sessions,
        "runs_per_session": max(1, int(-(-len(runs) // sessions))),
        "note": "Estimate only; replace seconds_per_epoch with the measured pilot value.",
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    report: dict[str, Any] = {
        "protocol": protocol_summary(),
        "code_fingerprint": code_fingerprint(args.root),
        "expected_counts": expected_counts(),
    }
    if args.manifest:
        report["manifest"] = validate_manifest(read_manifest(args.manifest))
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


def _cmd_evaluate_test(args: argparse.Namespace) -> int:
    """Guarded test-set evaluation. Requires a frozen selection file."""
    if not args.confirm_frozen:
        raise SystemExit(
            "Refusing to touch the test set: pass --confirm-frozen after freezing "
            "the validation-based selection."
        )
    selection_path = Path(args.selection)
    if not selection_path.exists():
        raise SystemExit(f"selection file not found: {selection_path}")
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", payload.get("selection", payload))
    if not isinstance(entries, list) or not entries:
        raise SystemExit("selection must be a non-empty list (or contain 'entries')")

    run_root = Path(args.run_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for entry in entries:
        run_id = entry["run_id"]
        stage = entry.get("stage", args.stage)
        variant = entry.get("variant", "W")
        run_dir = run_root / stage / run_id
        final_path = run_dir / "final.pt"
        if not final_path.exists():
            raise SystemExit(f"missing final artifact: {final_path}")
        artifact = torch.load(final_path, map_location="cpu", weights_only=False)
        if variant not in artifact:
            raise SystemExit(f"variant {variant!r} not present in {final_path}")
        config = RunConfig.from_dict(artifact["config"])
        model = MLP(config.dims)
        model.load_state_dict(artifact[variant])
        bundle = load_bundle(config.dataset, split="main", processed_dir=args.data_dir)
        metrics = evaluate_model(
            model,
            bundle.test_x,
            bundle.test_y,
            device=args.device,
        )
        weights = [layer.weight.detach().clone() for layer in model.layers]
        tau_final = artifact.get("tau_final")
        record = {
            "run_id": run_id,
            "stage": stage,
            "variant": variant,
            "dataset": config.dataset,
            "architecture": config.architecture,
            "seed": config.seed,
            "method": config.method,
            "alpha": config.alpha,
            "reg_lambda": config.reg_lambda,
            "tau_final": tau_final,
            "test": metrics,
            "omega": omega([weight.cpu().numpy() for weight in weights]),
            "layers": [spectral_summary(weight.cpu().numpy()) for weight in weights],
            "feasibility": (
                feasibility_report(weights, float(tau_final)) if tau_final is not None else None
            ),
        }
        records.append(record)

    (out_dir / "test_evaluation.json").write_text(
        json.dumps({"records": records}, indent=2, sort_keys=True), encoding="utf-8"
    )
    fieldnames = [
        "run_id",
        "stage",
        "variant",
        "dataset",
        "architecture",
        "seed",
        "method",
        "alpha",
        "reg_lambda",
        "tau_final",
        "test_loss",
        "test_accuracy",
        "omega",
        "violation",
        "feasible",
    ]
    with (out_dir / "test_evaluation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            feasibility = record["feasibility"] or {}
            writer.writerow(
                {
                    **{key: record.get(key) for key in fieldnames if key in record},
                    "test_loss": record["test"]["loss"],
                    "test_accuracy": record["test"]["accuracy"],
                    "violation": feasibility.get("violation"),
                    "feasible": feasibility.get("feasible"),
                }
            )
    print(
        json.dumps(
            {
                "status": "evaluated",
                "n_records": len(records),
                "output": str(out_dir),
                "note": "Test evaluated once against the frozen selection.",
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ram-experiments",
        description=f"Protocol {PROTOCOL_VERSION}: graph-energy bounds and nuclear-norm budgets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("manifest", help="Generate a frozen run manifest")
    manifest_parser.add_argument("--stage", choices=("pilot", "main"), required=True)
    manifest_parser.add_argument("--out", required=True)
    manifest_parser.set_defaults(func=_cmd_manifest)

    inventory_parser = subparsers.add_parser("inventory", help="Validate and summarize a manifest")
    inventory_parser.add_argument("--manifest", required=True)
    inventory_parser.set_defaults(func=_cmd_inventory)

    prepare_parser = subparsers.add_parser(
        "prepare-data", help="Build deterministic dataset splits"
    )
    prepare_parser.add_argument("--datasets", default="mnist,fashion_mnist")
    prepare_parser.add_argument("--raw-dir", default="data/raw")
    prepare_parser.add_argument("--processed-dir", default="data/processed")
    prepare_parser.add_argument("--no-download", action="store_true")
    prepare_parser.set_defaults(func=_cmd_prepare_data)

    run_parser = subparsers.add_parser("run", help="Execute a manifest shard with resume")
    run_parser.add_argument("--manifest", required=True)
    run_parser.add_argument("--stage", default=None)
    run_parser.add_argument("--run-root", default="runs")
    run_parser.add_argument("--data-dir", default="data/processed")
    run_parser.add_argument("--device", default="auto")
    run_parser.add_argument("--shard", type=int, default=0)
    run_parser.add_argument("--total-shards", type=int, default=1)
    run_parser.add_argument("--max-seconds", type=float, default=None)
    run_parser.add_argument("--limit", type=int, default=None)
    resume_group = run_parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume", dest="resume", action="store_true", default=True, help="Reanudar (por defecto)"
    )
    resume_group.add_argument(
        "--no-resume", dest="resume", action="store_false", help="Empezar de cero"
    )
    run_parser.add_argument("--allow-test", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.set_defaults(func=_cmd_run)

    verify_parser = subparsers.add_parser("verify", help="Run E0/E1 checks")
    verify_parser.add_argument("--suite", default="E0,E1")
    verify_parser.add_argument("--root", default=".")
    verify_parser.add_argument("--out", default="E0")
    verify_parser.set_defaults(func=_cmd_verify)

    analyze_parser = subparsers.add_parser(
        "analyze", help="Aggregate final metrics and calibration"
    )
    analyze_parser.add_argument("--manifest", required=True)
    analyze_parser.add_argument("--run-root", default="runs")
    analyze_parser.add_argument("--out", default="analysis")
    analyze_parser.set_defaults(func=_cmd_analyze)

    export_parser = subparsers.add_parser("export", help="Archive run directories")
    export_parser.add_argument("--manifest", required=True)
    export_parser.add_argument("--run-root", default="runs")
    export_parser.add_argument("--out", default="dist/ram_runs.zip")
    export_parser.set_defaults(func=_cmd_export)

    kaggle_parser = subparsers.add_parser("kaggle-plan", help="Estimate Kaggle sessions")
    kaggle_parser.add_argument("--manifest", required=True)
    kaggle_parser.add_argument("--seconds-per-epoch", type=float, required=True)
    kaggle_parser.add_argument("--session-seconds", type=float, default=32_400)
    kaggle_parser.set_defaults(func=_cmd_kaggle_plan)

    audit_parser = subparsers.add_parser("audit", help="Print protocol and repository audit")
    audit_parser.add_argument("--manifest", default=None)
    audit_parser.add_argument("--root", default=".")
    audit_parser.set_defaults(func=_cmd_audit)

    test_parser = subparsers.add_parser("evaluate-test", help="Guarded final test evaluation")
    test_parser.add_argument("--selection", required=True)
    test_parser.add_argument("--run-root", default="runs")
    test_parser.add_argument("--data-dir", default="data/processed")
    test_parser.add_argument("--out", default="analysis/test")
    test_parser.add_argument("--device", default="cpu")
    test_parser.add_argument("--stage", default="E2")
    test_parser.add_argument("--confirm-frozen", action="store_true")
    test_parser.set_defaults(func=_cmd_evaluate_test)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
