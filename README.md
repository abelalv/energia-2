<div align="center">

# RAM Graph Energy

**Implementación y benchmark del presupuesto global de norma nuclear en MLPs**
(protocolo `RAM-2026-10-13-v1`), listo para ejecutarse en Kaggle.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Managed with uv](https://img.shields.io/badge/managed%20with-uv-de5fe9.svg)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=precommit&logoColor=white)](./.pre-commit-config.yaml)
[![Kaggle](https://img.shields.io/badge/Kaggle-ready-20BEFF?logo=kaggle&logoColor=white)](./kaggle/README.md)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-fe5196.svg)](https://conventionalcommits.org)

</div>

---

## 📖 Qué es este repositorio

Paquete de trabajo para implementar y ejecutar el protocolo experimental del artículo
**“Graph Energy Bounds and Global Nuclear-Norm Budgets in Fully Connected Neural
Networks”** (*Results in Applied Mathematics*). Contiene:

- el **paquete `ram_experiments`** con la proyección nuclear conjunta, Ω(W), E(A), los
  modelos 3L/6L/9L, los métodos STD/WD/NUC/SN/ADMM/PSGD-M, el runner con reanudación y el
  manifiesto de 610 runs (540 E2 + 60 E4 + 10 continuaciones);
- las **pruebas E0/E1** y los 227 checks matemáticos preservados del paquete original;
- los **notebooks de Kaggle** y las utilidades para datasets, shards y presupuesto de
  sesión;
- la documentación original de traspaso, el manuscrito revisado y el material histórico
  para auditoría.

> ⚠️ **No hay resultados de benchmark en este repositorio.** No se ha entrenado en MNIST
> ni Fashion-MNIST. El código del piloto y del benchmark está listo, pero su ejecución
> requiere datos y GPU. Ninguna tabla o figura debe presentarse como resultado hasta
> completar E0/E1, el piloto y el benchmark congelado.

La verdad técnica central —stack, estructura, decisiones y reglas— vive en
[`CONSTITUTION.md`](./CONSTITUTION.md). El paquete original de traspaso empieza en
[`START_HERE.md`](./START_HERE.md).

## 📑 Tabla de contenidos

- [Estado del proyecto](#-estado-del-proyecto)
- [Stack](#-stack)
- [Primeros pasos locales](#-primeros-pasos-locales)
- [Ejecución en Kaggle](#-ejecución-en-kaggle)
- [Comandos del protocolo](#-comandos-del-protocolo)
- [Estructura](#-estructura)
- [Fronteras científicas](#-fronteras-científicas)
- [Documentación](#-documentación)
- [Licencia](#-licencia)

## 🚧 Estado del proyecto

| Área | Estado |
|------|:------:|
| Estructura, `uv`, Ruff, pre-commit, Makefile | ✅ |
| Núcleo matemático y proyección conjunta | ✅ |
| E0/E1 analíticos y oráculo CVXPY | ✅ |
| Modelos, métodos y bucle de entrenamiento | ✅ |
| Manifiesto 36 / 610 runs y deduplicación | ✅ |
| Reanudación checkpoint ↔ run continuo | ✅ |
| Notebooks y metadatos de Kaggle | ✅ (generados, no ejecutados en Kaggle) |
| Piloto de 36 runs | ⏳ |
| E2–E4 (600 runs + 10 continuaciones) | ⏳ |
| E3/E5/E6 (figuras, compresión, análisis final) | 🟡 |
| Evaluación final de test | 🔒 (bloqueada por diseño) |

Detalle y límites: [`docs/IMPLEMENTATION.md`](./docs/IMPLEMENTATION.md).

## 🛠️ Stack

| Categoría | Tecnología |
|-----------|------------|
| Lenguaje | Python 3.11+ |
| Entornos y dependencias | [`uv`](https://docs.astral.sh/uv/) (lockfile versionado) |
| Build backend | `uv_build` (src-layout) |
| Cómputo | PyTorch 2.x (CPU local; GPU en Kaggle) |
| Calidad | Ruff, pytest, pre-commit |
| Notebooks | Jupyter + `nbstripout` |
| Ejecución remota | Kaggle Notebooks (GPU T4 x2 / P100) |

## 🚀 Primeros pasos locales

```bash
git clone git@github.com:abelalv/energia-2.git
cd energia-2

uv sync
uv run pre-commit install      # o: make dev
make check                     # lint + formato + pruebas
make e0                        # E0/E1 + informe en E0/
```

> Regla del proyecto: toda gestión de entornos pasa por `uv`. La única excepción es
> Kaggle/Colab, donde se instala el paquete con `pip install -e . --no-deps` para no
> reemplazar el `torch` de la imagen.

### Datos

```bash
uv run python -m ram_experiments.cli prepare-data --datasets mnist,fashion_mnist
```

Genera los splits deterministas en `data/processed/` (ignorado por Git) con índices,
media/desviación y hashes. Ver [`data/README.md`](./data/README.md).

## ☁️ Ejecución en Kaggle

| Notebook | GPU | Propósito |
|----------|:---:|-----------|
| `kaggle/01_e0_e1/` | No | Puerta E0/E1; no sigas si falla. |
| `kaggle/02_pilot/` | Sí | 36 runs de 5 épocas; mide tiempos y memoria. |
| `kaggle/03_main_session/` | Sí | Shard de E2/E4 con `--max-seconds` y reanudación. |

```bash
# Publicar (ajusta el id de cada kernel-metadata.json)
kaggle kernels push -p kaggle/01_e0_e1
kaggle kernels push -p kaggle/02_pilot
kaggle kernels push -p kaggle/03_main_session
```

Guía completa: [`kaggle/README.md`](./kaggle/README.md) y [`docs/kaggle.md`](./docs/kaggle.md).

## 🧪 Comandos del protocolo

```bash
# Auditoría del protocolo y del manifiesto
uv run python -m ram_experiments.cli audit --manifest configs/main_manifest.jsonl

# Manifiestos
uv run python -m ram_experiments.cli manifest --stage pilot --out configs/pilot_manifest.jsonl
uv run python -m ram_experiments.cli manifest --stage main  --out configs/main_manifest.jsonl
uv run python -m ram_experiments.cli inventory --manifest configs/main_manifest.jsonl

# E0/E1
uv run python -m ram_experiments.cli verify --suite E0,E1 --out E0

# Ejecución acotada (local o Kaggle)
uv run python -m ram_experiments.cli run \
  --manifest configs/pilot_manifest.jsonl --stage pilot \
  --run-root runs --max-seconds 1800 --resume --device cpu

# Agregación y presupuesto
uv run python -m ram_experiments.cli analyze --manifest configs/main_manifest.jsonl --out analysis/main
uv run python -m ram_experiments.cli kaggle-plan --manifest configs/main_manifest.jsonl \
  --seconds-per-epoch 12 --session-seconds 30600
```

Equivalencia con la interfaz propuesta en el protocolo:

| Protocolo | Implementado |
|-----------|--------------|
| `python -m ram_experiments.audit` | `ram_experiments.cli audit` |
| `...verify --suite E0,E1` | `...cli verify --suite E0,E1` |
| `...pilot --no-test` | `...cli run --manifest configs/pilot_manifest.jsonl --stage pilot` |
| `...inventory --deduplicate` | `...cli inventory --manifest ...` |
| `...run --stage references,calibration,E2,E4 --resume` | `...cli run --stage E2,E4 --resume` |
| `...analyze --stage E3,E5,E6 --no-test` | `...cli analyze` (parcial) |
| `...evaluate-test --selection` | `...cli evaluate-test --selection ... --confirm-frozen` |
| `...export --include-failures` | `...cli export` |

## 📂 Estructura

```text
energia-2/
├── src/ram_experiments/       # paquete: energía, proyección, modelos, métodos, runner, CLI
├── tests/                     # E0, E1, datos, manifiesto y entrenamiento smoke
├── configs/                   # snapshots del protocolo y manifiestos
├── kaggle/                    # notebooks + kernel-metadata.json
├── scripts/                   # generadores de notebooks y datasets de Kaggle
├── docs/                      # implementación, Kaggle, ADRs
├── notebooks/                 # exploración (no fuente de resultados)
├── data/                      # splits procesados (ignorados por Git)
├── figures/                   # figuras generadas por código (pendientes)
├── mathematical_checks/       # 227 checks preservados (no modificados)
├── manuscript/                # manuscrito revisado (PDF + LaTeX)
├── historical_reference/      # manuscrito histórico para auditoría
├── START_HERE.md              # entrada del paquete de traspaso original
├── MATHEMATICAL_IMPLEMENTATION_SPECIFICATION.md
├── EXPERIMENTAL_PROTOCOL.md
├── CLAIMS_AND_EVIDENCE_MATRIX.md
├── MATHEMATICAL_AUDIT.md
├── HISTORICAL_CODE_AUDIT.md
├── CONSTITUTION.md
├── pyproject.toml
└── uv.lock
```

## 🔬 Fronteras científicas

- **Ω(W) = 2 Σ‖W_l‖_* no es E(A)**; se reportan por separado. Los biases entran en el
  objetivo y en los diagnósticos de espacio completo, no en el presupuesto.
- El modelo primario de ADMM es **W** del último epoch completo; **Z** y **P(W)** son
  secundarios y se etiquetan como tales.
- La proyección es una aproximación numérica float64, no una cota rigurosa de redondeo.
- Los residuales finitos no son un teorema de convergencia; el rango umbralizado no es una
  cota algebraica rigurosa; el rango efectivo no es compresión medida.
- El test se evalúa **una vez**, después de congelar la selección por validación.
- Los fallos, empates y derrotas de ADMM se reportan igual que las victorias.

## 📚 Documentación

| Documento | Para qué |
|-----------|----------|
| [`START_HERE.md`](./START_HERE.md) | Entrada del handoff original. |
| [`MATHEMATICAL_IMPLEMENTATION_SPECIFICATION.md`](./MATHEMATICAL_IMPLEMENTATION_SPECIFICATION.md) | Fórmulas, proyección, ADMM, certificados. |
| [`EXPERIMENTAL_PROTOCOL.md`](./EXPERIMENTAL_PROTOCOL.md) | E0–E6, inventario, fallos, análisis. |
| [`docs/IMPLEMENTATION.md`](./docs/IMPLEMENTATION.md) | Qué está implementado y qué falta. |
| [`CLAIMS_AND_EVIDENCE_MATRIX.md`](./CLAIMS_AND_EVIDENCE_MATRIX.md) | Redacción permitida según evidencia. |
| [`MATHEMATICAL_AUDIT.md`](./MATHEMATICAL_AUDIT.md) / [`HISTORICAL_CODE_AUDIT.md`](./HISTORICAL_CODE_AUDIT.md) | Correcciones M01–M28 y C01–C27. |
| [`manuscript/`](./manuscript/) | Manuscrito revisado (PDF y LaTeX). |
| [`historical_reference/`](./historical_reference/) | Material histórico, solo procedencia. |

## 📄 Licencia

MIT. Ver [`LICENSE`](./LICENSE). Los datasets conservan sus licencias originales
(ver [`data/README.md`](./data/README.md)).

---

<div align="center">
<sub>Hecho con ❤️ por abel alvarez · protocolo RAM-2026-10-13-v1</sub>
</div>
