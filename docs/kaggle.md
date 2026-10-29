# Kaggle — guía rápida

La guía operativa completa está en [`../kaggle/README.md`](../kaggle/README.md). Resumen:

1. **E0/E1** (`kaggle/01_e0_e1/`), sin GPU. Es una puerta bloqueante.
2. **Piloto** (`kaggle/02_pilot/`), GPU, 36 runs de 5 épocas, seed 42. Mide tiempos.
3. **Benchmark** (`kaggle/03_main_session/`), GPU, shards de E2/E4 con `--max-seconds`.

## Comandos equivalentes en local

```bash
# Verificaciones
make e0

# Datos (una vez)
uv run python -m ram_experiments.cli prepare-data --datasets mnist,fashion_mnist

# Manifiestos
uv run python -m ram_experiments.cli manifest --stage pilot --out configs/pilot_manifest.jsonl
uv run python -m ram_experiments.cli manifest --stage main  --out configs/main_manifest.jsonl
uv run python -m ram_experiments.cli inventory --manifest configs/main_manifest.jsonl

# Ejecución acotada
uv run python -m ram_experiments.cli run \
  --manifest configs/pilot_manifest.jsonl --stage pilot \
  --run-root runs --max-seconds 1800 --resume --device cpu
```

## Reglas de Kaggle

- `pip install -e . --no-deps`: se conserva el `torch`/`torchvision` de la imagen de
  Kaggle. La excepción a "solo `uv`" está documentada en `CONSTITUTION.md`.
- No se usa el test para elegir hiperparámetros, épocas ni figuras.
- Cada sesión guarda `/kaggle/working/runs`; para reanudar, se sube como dataset privado.
- La estimación de sesiones se obtiene con `ram_experiments.cli kaggle-plan` usando el
  tiempo por época medido en el piloto.

## Limitación

Los notebooks se generaron y validaron sintácticamente, pero **no se ejecutaron en
Kaggle** desde esta máquina. La primera ejecución debe registrar la versión de Python,
PyTorch, CUDA, GPU y VRAM en `environment.json` de cada run.
