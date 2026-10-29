"""Run a manifest on two GPUs as two independent, disjoint workers.

This does **not** use ``DataParallel``: each worker trains different runs with the
protocol's original batch size, seeds and numerical profile. Sharing one model
across GPUs would change the effective batch and is not part of
RAM-2026-10-13-v1.

Typical Kaggle session (T4 x2)::

    python scripts/run_dual_gpu.py --stage E2 --max-seconds 30600

The script first pre-computes the STD references on both GPUs, then splits the
remaining runs round-robin between the two workers. Each worker gets its own
``CUDA_VISIBLE_DEVICES`` and writes to the shared ``runs/`` directory.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _build_command(
    *,
    manifest: str,
    stage: str,
    run_root: str,
    data_dir: str,
    max_seconds: float,
    shard: int,
    total_shards: int,
    methods: str | None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "ram_experiments.cli",
        "run",
        "--manifest",
        manifest,
        "--stage",
        stage,
        "--run-root",
        run_root,
        "--data-dir",
        data_dir,
        "--device",
        "cuda:0",
        "--shard",
        str(shard),
        "--total-shards",
        str(total_shards),
        "--max-seconds",
        str(max_seconds),
        "--resume",
    ]
    if methods:
        command.extend(["--methods", methods])
    return command


def _launch(command: list[str], gpu: str, log_path: Path, dry_run: bool) -> subprocess.Popen | None:
    print(f"[gpu {gpu}] {' '.join(command)}")
    print(f"[gpu {gpu}] log: {log_path}")
    if dry_run:
        return None
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment["PYTHONUNBUFFERED"] = "1"
    environment.setdefault("OMP_NUM_THREADS", "2")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # Keep the handle alive until the process finishes.
    process._ram_log_handle = handle  # type: ignore[attr-defined]
    return process


def _wait(processes: list[subprocess.Popen], logs: list[Path]) -> list[int]:
    codes: list[int] = []
    for process, log_path in zip(processes, logs, strict=True):
        code = process.wait()
        handle = getattr(process, "_ram_log_handle", None)
        if handle is not None:
            handle.close()
        codes.append(code)
        print(f"\n===== fin gpu (código {code}): {log_path} =====")
        if log_path.exists():
            tail = deque(
                log_path.read_text(encoding="utf-8", errors="replace").splitlines(), maxlen=25
            )
            print("\n".join(tail))
    return codes


def _phase(
    *,
    label: str,
    gpus: list[str],
    command_kwargs: dict,
    log_dir: Path,
    dry_run: bool,
) -> int:
    processes: list[subprocess.Popen] = []
    logs: list[Path] = []
    for index, gpu in enumerate(gpus):
        command = _build_command(
            **command_kwargs,
            shard=index,
            total_shards=len(gpus),
        )
        log_path = log_dir / f"{label}-gpu{gpu}.log"
        process = _launch(command, gpu, log_path, dry_run)
        if process is not None:
            processes.append(process)
            logs.append(log_path)
    if dry_run:
        return 0
    codes = _wait(processes, logs)
    return 0 if all(code == 0 for code in codes) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/main_manifest.jsonl")
    parser.add_argument("--stage", choices=("pilot", "E2", "E4"), default="E2")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--max-seconds", type=float, default=30_600)
    parser.add_argument("--gpus", default="0,1", help="IDs físicos separados por coma")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="No precalcular STD (solo si ya están completos)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise SystemExit("--gpus no puede estar vacío")
    if len(gpus) > 2:
        raise SystemExit("este runner está pensado para 1 o 2 GPUs")

    manifest = Path(args.manifest)
    if not manifest.exists():
        raise SystemExit(f"manifest no encontrado: {manifest}")
    log_dir = Path(args.log_dir)
    session_started = time.monotonic()

    common = {
        "manifest": str(manifest),
        "stage": args.stage,
        "run_root": args.run_root,
        "data_dir": args.data_dir,
        "max_seconds": args.max_seconds,
    }

    if args.stage in {"pilot", "E2"} and not args.skip_prepare:
        print("Fase 1/2: referencias STD en paralelo")
        code = _phase(
            label=f"{args.stage}-std",
            gpus=gpus,
            command_kwargs={**common, "methods": "STD"},
            log_dir=log_dir,
            dry_run=args.dry_run,
        )
        if code != 0:
            print("Fase STD falló; no se lanza el benchmark.")
            return code
    else:
        print("Fase 1/2: referencias STD omitidas")

    if args.stage == "E4" and not (Path(args.run_root) / "E2").exists():
        print(
            "ADVERTENCIA: no existe runs/E2. E4 necesita E2 completo (STD y ADMM 9L padre).",
            file=sys.stderr,
        )

    remaining = args.max_seconds - (time.monotonic() - session_started)
    if not args.dry_run and remaining <= 0:
        print("No queda presupuesto de sesión después de las referencias STD.")
        return 1

    print("Fase 2/2: benchmark en paralelo")
    code = _phase(
        label=args.stage,
        gpus=gpus,
        command_kwargs={**common, "methods": None, "max_seconds": max(0.0, remaining)},
        log_dir=log_dir,
        dry_run=args.dry_run,
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
