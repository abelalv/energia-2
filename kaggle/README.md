# Kaggle — ejecución del protocolo RAM-2026-10-13-v1

Los notebooks de esta carpeta son **orquestadores finos**: instalan el paquete, preparan
los datos y llaman a `ram_experiments`. La lógica científica vive en `src/ram_experiments/`,
no en el notebook.

## Orden obligatorio

| # | Notebook | GPU | Qué hace | Puerta |
|---|----------|:---:|----------|--------|
| 1 | `01_e0_e1/01_e0_e1.ipynb` | No | E0/E1 + 227 checks matemáticos | **Bloqueante**. No sigas si falla. |
| 2 | `02_pilot/02_pilot.ipynb` | Sí | 36 runs de 5 épocas, seed 42 | Mide tiempos/memoria; no es benchmark. |
| 3 | `03_main_session/03_main_session.ipynb` | Sí | Shard de E2/E4 con `--max-seconds` | Requiere presupuesto acordado. |

## 1. Publicar los notebooks con la CLI de Kaggle

```bash
pip install kaggle            # o usa la imagen de Kaggle
export KAGGLE_USERNAME=...
export KAGGLE_KEY=...

kaggle kernels push -p kaggle/01_e0_e1
kaggle kernels push -p kaggle/02_pilot
kaggle kernels push -p kaggle/03_main_session
```

Antes de publicar, cambia `id` en cada `kernel-metadata.json` si tu usuario no es
`abelalv`. Los notebooks también se pueden subir a mano a Kaggle.

## 2. Código: Git o dataset

Por defecto el notebook clona `https://github.com/abelalv/energia-2.git` con internet
activado. Si el repo es privado o la sesión no tiene internet:

1. Ejecuta en local:
   ```bash
   uv run python scripts/build_kaggle_assets.py --out dist/kaggle
   ```
2. Sube `dist/kaggle/code` como **Kaggle Dataset** privado.
3. Añade el dataset al notebook (`Add Data`) y deja `enable_internet: false` si lo deseas.
   El notebook busca cualquier `pyproject.toml` con `src/ram_experiments/` y copia el repo
   a `/kaggle/working/energia-2`.

## 3. Datos: descarga o dataset

Opción A (recomendada para la primera vez): internet activado y el notebook ejecuta

```bash
python -m ram_experiments.cli prepare-data --datasets mnist,fashion_mnist
```

Descarga MNIST y Fashion-MNIST por `torchvision`, fija el split determinista
(`PCG64(20261013)`), calcula media/desviación solo con el train principal y escribe
`data/processed/{dataset}_{main,pilot}.npz` con hashes.

Opción B (reproducible y sin red): sube `dist/kaggle/data` como dataset y móntalo. El
notebook busca `**/data/processed/mnist_main.npz` y usa ese directorio.

> Los datos **no** se versionan en Git. Los `.npz` procesados se pueden publicar como
> dataset privado de Kaggle; documenta la licencia y la fecha de descarga.

## 4. Dos GPUs en paralelo (T4 x2)

El notebook `03_main_session` incluye una celda **“Ejecución en 2 GPUs”**. No usa
`DataParallel`: lanza **dos procesos independientes**, cada uno con su
`CUDA_VISIBLE_DEVICES` y un shard disjunto del manifiesto. Así cada run conserva
batch 64, semillas y perfil numérico del protocolo.

El script hace dos fases:

1. **Referencias STD:** reparte los runs `STD` entre las dos GPUs.
2. **Resto del manifiesto:** reparte round-robin los demás runs entre las dos GPUs.

```bash
python scripts/run_dual_gpu.py \
  --manifest configs/main_manifest.jsonl \
  --stage E2 \
  --data-dir /kaggle/working/data/processed \
  --run-root /kaggle/working/runs \
  --max-seconds 30600 \
  --gpus 0,1
```

Reglas:

- No ejecutes a la vez la celda de 2 GPUs y la de 1 GPU.
- Para E4, E2 debe estar completo (STD y el ADMM 9L padre).
- Los logs quedan en `logs/E2-gpu0.log` y `logs/E2-gpu1.log`.
- Cada worker escribe runs distintos en `runs/`; los resúmenes de sesión llevan PID
  para no colisionar.
- Si una GPU falla, el script termina con código distinto de cero; revisa el log.
- La reanudación es la misma: `--resume` reutiliza runs completos y checkpoints.

## 5. Sesiones, tiempo y reanudación

- GPU de Kaggle: T4 x2 / P100, 12 h por sesión. Usa `RAM_MAX_SECONDS=30600` (8 h 30 min)
  para dejar margen de guardado; el runner corta en un límite de época y guarda
  `checkpoint.pt`.
- La cuota semanal de GPU es limitada. El benchmark principal son **600 runs de 25
  épocas + 10 continuaciones de 75 épocas**; planifica shards y sesiones.
- Para reanudar: descarga `/kaggle/working/runs`, súbelo como dataset privado y copia los
  run directories incompletos a `/kaggle/working/runs`. Vuelve a lanzar con `--resume`
  (el notebook ya lo usa).
- No ejecutes dos sesiones sobre el mismo `run_id`: el runner reutiliza runs completos,
  pero dos procesos concurrentes pueden corromper artefactos.

## 6. Presupuesto medido antes del benchmark

Después del piloto, ejecuta en local o en el notebook:

```bash
python -m ram_experiments.cli kaggle-plan \
  --manifest configs/main_manifest.jsonl \
  --seconds-per-epoch <medido> \
  --session-seconds 30600 \
  --gpus 2 --efficiency 0.85
```

Sustituye `<medido>` por el tiempo real por época del piloto para la celda
arquitectura/método/dataset correspondiente. El resultado es una **estimación**, no una
garantía. Añade 20 % de margen operativo y 10 % para reintentos de infraestructura.

## 7. Reglas que el notebook no relaja

- E0/E1 antes de cualquier entrenamiento.
- Sin acceso al test hasta cerrar la selección por validación.
- Semillas 42, 123, 456, 789, 1024; no se descartan semillas por accuracy.
- WD/NUC: una sola vez por candidato (no por alpha); SN/STD tampoco se triplican.
- ADMM: modelo primario W del último epoch; Z y P(W) son secundarios.
- Los fallos se reportan; no se imputan ni se ocultan.

## 8. Limitaciones de esta adaptación

- Los notebooks no se han ejecutado en Kaggle desde esta máquina; se validaron el paquete,
  los manifiestos y las pruebas E0/E1 en CPU.
- El paquete instala `torch`/`torchvision` de Kaggle con `pip install -e . --no-deps`.
  La versión observada debe guardarse en `environment.json` de cada run.
- La proyección conjunta se ejecuta en CPU/float64 por defecto (`projection_device`),
  mientras el entrenamiento usa la GPU. Cambiar ese perfil exige un nuevo hash y E0.
