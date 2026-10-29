# notebooks/

**EN** — Exploration and prototyping. Notebooks are drafts: once a function works,
move it to `src/ram_experiments/` and import it from here. Outputs are stripped
automatically by `nbstripout` before each commit, so the repo stays light and diffs stay
readable.

**ES** — Exploración y prototipos. Los notebooks son borradores: cuando una función
funciona, se mueve a `src/ram_experiments/` y se importa desde aquí. `nbstripout`
elimina las salidas antes de cada commit para mantener el repo liviano y los diffs legibles.

## Convención de nombres / Naming

`NN_tema-corto.ipynb` → `01_eda-datos.ipynb`, `02_pinn-baseline.ipynb`, `03_figuras-paper.ipynb`

## Local (uv)

```bash
make notebooks          # uv sync --group notebooks + registra el kernel
```

## Kaggle (vía principal para el benchmark)

Los notebooks de ejecución viven en `kaggle/`, no aquí, porque Kaggle necesita su propio
`kernel-metadata.json` y un `MAX_SECONDS` explícito:

- `kaggle/01_e0_e1/` — puerta E0/E1, sin GPU.
- `kaggle/02_pilot/` — 36 runs de 5 épocas, GPU.
- `kaggle/03_main_session/` — shards de E2/E4, GPU, reanudable.

Guía: [`../kaggle/README.md`](../kaggle/README.md) y [`../docs/kaggle.md`](../docs/kaggle.md).

## Google Colab

Colab trae su propio Python y PyTorch/JAX con CUDA; **no** reemplaces ese entorno con
`uv sync`. Instala solo tu paquete encima / Install only your package on top:

```python
# Celda 1 — clonar e instalar el paquete del proyecto (sin tocar torch/jax de Colab)
!git clone https://github.com/abelalv/energia-2.git
%cd energia-2
!pip install -q -e . --no-deps
```

> Excepción documentada a la regla "solo uv": dentro de Colab/Kaggle se usa `pip` porque el
> entorno no es nuestro. Ver `references/colab.md` del skill.

> Los notebooks de `kaggle/` son los únicos que ejecutan el protocolo; los de esta carpeta
> son exploración y no deben convertirse en la fuente de resultados del paper.
