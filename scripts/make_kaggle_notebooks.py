"""Generate the three Kaggle notebooks from a single source of truth.

Run with ``uv run python scripts/make_kaggle_notebooks.py``. The notebooks are
thin orchestrators: they install the package, prepare data and call the CLI.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "kaggle"

SETUP = r"""import os, pathlib, shutil, subprocess, sys

REPO_URL = "https://github.com/abelalv/energia-2.git"
REPO_DIR = pathlib.Path("/kaggle/working/energia-2")

if not (REPO_DIR / "pyproject.toml").exists():
    candidates = [
        path
        for path in pathlib.Path("/kaggle/input").rglob("pyproject.toml")
        if (path.parent / "src" / "ram_experiments").exists()
    ]
    if candidates:
        shutil.copytree(candidates[0].parent, REPO_DIR, dirs_exist_ok=True)
        print("repo copiado desde", candidates[0].parent)
    else:
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)

os.chdir(REPO_DIR)
# Los subprocesos del notebook deben encontrar el paquete también sin instalación editable.
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(REPO_DIR / "src"), os.environ.get("PYTHONPATH", "")]
)
try:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"],
        check=True,
    )
except subprocess.CalledProcessError:
    # Kaggle sin red o sin uv_build: usar src/ directamente. El entorno sigue siendo el de Kaggle.
    sys.path.insert(0, str(REPO_DIR / "src"))
    print("pip install -e falló; usando src/ en sys.path")
print("cwd:", pathlib.Path.cwd())
"""

ENVIRONMENT = r"""import platform, torch

print("python:", platform.python_version())
print("torch:", torch.__version__)
print("cuda disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("vram GiB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))

# Protocol RAM-2026-10-13-v1: float32 without AMP/TF32.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
print("tf32:", torch.backends.cuda.matmul.allow_tf32)
"""

DATA = r"""import os, pathlib, subprocess, sys

DATA_DIR = pathlib.Path(os.environ.get("RAM_DATA_DIR", "/kaggle/working/data/processed"))

if not (DATA_DIR / "mnist_main.npz").exists():
    attached = sorted(pathlib.Path("/kaggle/input").rglob("mnist_main.npz"))
    if attached:
        DATA_DIR = attached[0].parent
        print("datos adjuntos:", DATA_DIR)
    else:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ram_experiments.cli",
                "prepare-data",
                "--datasets",
                "mnist,fashion_mnist",
                "--raw-dir",
                "/kaggle/working/data/raw",
                "--processed-dir",
                str(DATA_DIR),
            ],
            check=True,
        )

print("data dir:", DATA_DIR)
print(sorted(path.name for path in DATA_DIR.glob("*.npz")))
print("define RAM_DATA_DIR en las celdas de ejecución:", DATA_DIR)
"""

E0 = r"""import subprocess, sys

subprocess.run(
    [
        sys.executable,
        "-m",
        "ram_experiments.cli",
        "verify",
        "--suite",
        "E0,E1",
        "--out",
        "/kaggle/working/E0",
    ],
    check=True,
)
subprocess.run([sys.executable, "mathematical_checks/verify_mathematics.py"], check=True)
"""

PILOT = r"""import pathlib, subprocess, sys

MANIFEST = pathlib.Path("configs/pilot_manifest.jsonl")
RUN_ROOT = "/kaggle/working/runs"
MAX_SECONDS = 30_000  # 8 h 20 min; deja margen para guardar la sesión

subprocess.run(
    [
        sys.executable,
        "-m",
        "ram_experiments.cli",
        "manifest",
        "--stage",
        "pilot",
        "--out",
        str(MANIFEST),
    ],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "ram_experiments.cli", "inventory", "--manifest", str(MANIFEST)],
    check=True,
)
subprocess.run(
    [
        sys.executable,
        "-m",
        "ram_experiments.cli",
        "run",
        "--manifest",
        str(MANIFEST),
        "--stage",
        "pilot",
        "--run-root",
        RUN_ROOT,
        "--data-dir",
        str(DATA_DIR),
        "--device",
        "cuda",
        "--max-seconds",
        str(MAX_SECONDS),
        "--resume",
    ],
    check=True,
)
"""

PILOT_ANALYSIS = r"""import pathlib, subprocess, sys

subprocess.run(
    [
        sys.executable,
        "-m",
        "ram_experiments.cli",
        "analyze",
        "--manifest",
        "configs/pilot_manifest.jsonl",
        "--run-root",
        "/kaggle/working/runs",
        "--out",
        "/kaggle/working/analysis/pilot",
    ],
    check=True,
)
"""

CONFIG = r"""import os, pathlib, subprocess, sys

STAGE = os.environ.get("RAM_STAGE", "E2")          # E2 o E4
SHARD = int(os.environ.get("RAM_SHARD", "0"))      # solo para 1 GPU
TOTAL_SHARDS = int(os.environ.get("RAM_TOTAL_SHARDS", "1"))  # solo para 1 GPU
MAX_SECONDS = int(os.environ.get("RAM_MAX_SECONDS", "30600"))  # 8 h 30 min

MANIFEST = pathlib.Path("configs/main_manifest.jsonl")
RUN_ROOT = "/kaggle/working/runs"

if not MANIFEST.exists():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ram_experiments.cli",
            "manifest",
            "--stage",
            "main",
            "--out",
            str(MANIFEST),
        ],
        check=True,
    )
subprocess.run(
    [sys.executable, "-m", "ram_experiments.cli", "inventory", "--manifest", str(MANIFEST)],
    check=True,
)
print("STAGE:", STAGE, "MAX_SECONDS:", MAX_SECONDS)
"""

DUAL = r"""import subprocess, sys

# Dos procesos independientes, uno por GPU. No usa DataParallel: cada run conserva
# batch 64, semillas y perfil numérico del protocolo.
subprocess.run(
    [
        sys.executable,
        "scripts/run_dual_gpu.py",
        "--manifest",
        str(MANIFEST),
        "--stage",
        STAGE,
        "--run-root",
        RUN_ROOT,
        "--data-dir",
        str(DATA_DIR),
        "--max-seconds",
        str(MAX_SECONDS),
        "--gpus",
        "0,1",
    ],
    check=True,
)
"""

SINGLE = r"""import subprocess, sys

# Alternativa de una sola GPU. SHARD y TOTAL_SHARDS permiten repartir sesiones.
subprocess.run(
    [
        sys.executable,
        "-m",
        "ram_experiments.cli",
        "run",
        "--manifest",
        str(MANIFEST),
        "--stage",
        STAGE,
        "--run-root",
        RUN_ROOT,
        "--data-dir",
        str(DATA_DIR),
        "--device",
        "cuda:0",
        "--shard",
        str(SHARD),
        "--total-shards",
        str(TOTAL_SHARDS),
        "--max-seconds",
        str(MAX_SECONDS),
        "--resume",
    ],
    check=True,
)
"""

MAIN_ANALYSIS = r"""import subprocess, sys

subprocess.run(
    [
        sys.executable,
        "-m",
        "ram_experiments.cli",
        "analyze",
        "--manifest",
        "configs/main_manifest.jsonl",
        "--run-root",
        "/kaggle/working/runs",
        "--out",
        "/kaggle/working/analysis/main",
    ],
    check=True,
)
"""


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip("\n").splitlines()],
    }


def _markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip("\n").splitlines()],
    }


def _notebook(cells: list[dict]) -> dict:
    ordered = [{"id": str(index), **cell} for index, cell in enumerate(cells)]
    return {
        "cells": ordered,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build() -> None:
    notebooks = {
        "01_e0_e1/01_e0_e1.ipynb": _notebook(
            [
                _markdown(
                    "# RAM Graph Energy — E0/E1 (puerta bloqueante)\n\n"
                    "Este notebook **no entrena**. Comprueba que las fórmulas que se van a "
                    "optimizar, proyectar y medir coinciden con la especificación. "
                    "No continúes al piloto si este notebook falla."
                ),
                _markdown("## 1. Instalar el paquete"),
                _code(SETUP),
                _markdown("## 2. Entorno y GPU"),
                _code(ENVIRONMENT),
                _markdown("## 3. Verificaciones E0/E1 y checks matemáticos preservados"),
                _code(E0),
                _markdown(
                    "## 4. Lectura del informe\n\n"
                    "`/kaggle/working/E0/E0_E1_report.json` debe tener `status = passed`. "
                    "Los 227 checks de `mathematical_checks` complementan, no sustituyen, E0/E1."
                ),
                _code(
                    "import json, pathlib\n"
                    "path = pathlib.Path('/kaggle/working/E0/E0_E1_report.json')\n"
                    "report = json.loads(path.read_text())\n"
                    "print(report['status'], report['returncode'])"
                ),
            ]
        ),
        "02_pilot/02_pilot.ipynb": _notebook(
            [
                _markdown(
                    "# RAM Graph Energy — piloto de 36 runs\n\n"
                    "Un seed (42), ambos datasets, tres arquitecturas, seis métodos y cinco "
                    "épocas. El piloto mide tiempos y memoria; **no** es el benchmark principal. "
                    "No consulta el conjunto de test."
                ),
                _markdown("## 1. Instalar el paquete"),
                _code(SETUP),
                _markdown("## 2. Entorno"),
                _code(ENVIRONMENT),
                _markdown("## 3. Datos (descarga o dataset adjunto)"),
                _code(DATA),
                _markdown(
                    "## 4. Manifiesto, inventario y ejecución\n\n"
                    "`MAX_SECONDS=30000` deja margen dentro de la sesión GPU de Kaggle. "
                    "Si se agota, el runner guarda checkpoint y la próxima sesión reanuda."
                ),
                _code(PILOT),
                _markdown("## 5. Tabla de tiempos y métricas"),
                _code(PILOT_ANALYSIS),
                _markdown(
                    "## 6. Qué devolver a los autores\n\n"
                    "Tiempo por método/arquitectura/dataset, memoria pico, coste de evaluación, "
                    "checkpoints y E4. El benchmark principal **no** se lanza hasta acordar el "
                    "presupuesto medido."
                ),
            ]
        ),
        "03_main_session/03_main_session.ipynb": _notebook(
            [
                _markdown(
                    "# RAM Graph Energy — sesión del benchmark principal\n\n"
                    "Ejecuta un shard de E2/E4 con presupuesto de tiempo. Requiere E0/E1 "
                    "aprobados y el piloto medido. Para E4, E2 debe estar completo porque las "
                    "continuaciones 25→100 reutilizan el checkpoint padre."
                ),
                _markdown("## 1. Instalar el paquete"),
                _code(SETUP),
                _markdown("## 2. Entorno"),
                _code(ENVIRONMENT),
                _markdown("## 3. Datos"),
                _code(DATA),
                _markdown(
                    "## 4. Configuración de la sesión\n\n"
                    "Variables: `RAM_STAGE` (`E2` o `E4`) y `RAM_MAX_SECONDS`. "
                    "La ejecución en 2 GPUs reparte los runs automáticamente; no toques "
                    "`SHARD` ni `TOTAL_SHARDS` si usas esa opción."
                ),
                _code(CONFIG),
                _markdown(
                    "## 5. Ejecución en 2 GPUs (recomendado en T4 x2)\n\n"
                    "Lanza **dos procesos independientes**, uno por GPU, con shards disjuntos. "
                    "No es `DataParallel`: cada run conserva batch 64, semillas y perfil "
                    "numérico. Primero precalcula las referencias STD y después reparte el "
                    "resto del manifiesto."
                ),
                _code(DUAL),
                _markdown(
                    "## 6. Ejecución en 1 GPU (alternativa)\n\n"
                    "Úsala solo si no tienes dos GPUs o quieres repartir sesiones manualmente "
                    "con `SHARD` y `TOTAL_SHARDS`."
                ),
                _code(SINGLE),
                _markdown("## 7. Agregación"),
                _code(MAIN_ANALYSIS),
                _markdown(
                    "## 8. Reanudación entre sesiones\n\n"
                    "Sube la carpeta `runs/` como dataset privado y móntala en "
                    "`/kaggle/input/...`; copia solo los run directories incompletos a "
                    "`/kaggle/working/runs` y vuelve a ejecutar la celda de 2 GPUs. El script "
                    "usa `--resume`."
                ),
            ]
        ),
    }

    for relative, notebook in notebooks.items():
        path = KAGGLE / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(notebook, indent=1, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("escrito:", path.relative_to(ROOT))


if __name__ == "__main__":
    build()
