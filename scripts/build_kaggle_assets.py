"""Build the two optional Kaggle datasets: code snapshot and processed data."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CODE_IGNORE = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "dist",
    "runs",
    "E0",
    "E1",
    "analysis",
    "data",
    "figures",
    "manuscript",
    "historical_reference",
}


def copy_code(destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        ROOT,
        destination,
        ignore=shutil.ignore_patterns(*CODE_IGNORE, "*.pt", "*.npz", "*.pdf"),
    )
    metadata = {
        "title": "Energia-2 code",
        "id": "abelalv/energia-2-code",
        "licenses": [{"name": "MIT"}],
    }
    (destination / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def copy_data(destination: Path, processed_dir: Path) -> int:
    if destination.exists():
        shutil.rmtree(destination)
    target = destination / "data" / "processed"
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(processed_dir.glob("*.npz")):
        shutil.copy2(path, target / path.name)
        count += 1
    for path in sorted(processed_dir.glob("*_manifest.json")):
        shutil.copy2(path, target / path.name)
    metadata = {
        "title": "Energia-2 MNIST Fashion-MNIST processed",
        "id": "abelalv/energia-2-data",
        "licenses": [{"name": "CC-BY-SA-3.0"}, {"name": "MIT"}],
    }
    (destination / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="dist/kaggle")
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()

    out = Path(args.out)
    copy_code(out / "code")
    n_files = copy_data(out / "data", Path(args.processed_dir))
    print(
        json.dumps(
            {"code": str(out / "code"), "data": str(out / "data"), "npz_files": n_files},
            indent=2,
        )
    )
    print("Sube cada carpeta con: kaggle datasets create -p <carpeta>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
