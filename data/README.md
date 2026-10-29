# data/

**EN** — Local data. `data/raw/` and `data/processed/` are git-ignored. Version only small
samples or metadata. Document every source below so results can be reproduced.

**ES** — Datos locales. `data/raw/` y `data/processed/` están en `.gitignore`. Versiona solo
muestras pequeñas o metadatos. Documenta cada fuente para que los resultados sean
reproducibles.

| Dataset | Fuente / Source | Descarga | Licencia | Carpeta |
|---------|-----------------|----------|----------|---------|
| MNIST | `torchvision.datasets.MNIST` (LeCun et al.) | `prepare-data` | CC BY-SA 3.0 | `raw/`, `processed/` |
| Fashion-MNIST | `torchvision.datasets.FashionMNIST` (Xiao et al., Zalando) | `prepare-data` | MIT | `raw/`, `processed/` |

## Split del protocolo (RAM-2026-10-13-v1)

- Del **train oficial** (60 000): 50 000 train / 10 000 validation.
- Test oficial (10 000) reservado y **no** usado para selección.
- Split por clase con `numpy.random.Generator(PCG64(20261013))`; índices y hashes
  publicados por `prepare-data`.
- Normalización: media y desviación estándar calculadas **solo** con el train principal
  de cada dataset; se congelan en el `.npz`.
- Piloto: prefijos estratificados de 10 000 train / 2 000 validation del split anterior.
- Sin aumento de datos.

## Preparación

```bash
uv run python -m ram_experiments.cli prepare-data --datasets mnist,fashion_mnist
```

Genera:

```text
data/processed/
├── mnist_main.npz
├── mnist_pilot.npz
├── mnist_manifest.json
├── fashion_mnist_main.npz
├── fashion_mnist_pilot.npz
└── fashion_mnist_manifest.json
```

Cada `.npz` contiene `train_x`, `train_y`, `val_x`, `val_y`, `test_x`, `test_y`, `mean`,
`std`, `indices_train`, `indices_val` y `hashes`. Los tensores están aplanados
(`784` columnas) y normalizados en `float32`.

## Kaggle

Los `.npz` procesados se pueden publicar como **Kaggle Dataset privado** con
`scripts/build_kaggle_assets.py`. Los notebooks buscan
`/kaggle/input/**/data/processed/mnist_main.npz` y usan ese directorio sin volver a
descargar. Documenta la fecha de descarga y la versión de `torchvision` en el informe.

> No se versionan imágenes crudas, checkpoints ni resultados. Las figuras se generan desde
> código y viven en `figures/`.
