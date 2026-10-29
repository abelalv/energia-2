# Implementación — mapa, estado y límites

Protocolo: `RAM-2026-10-13-v1`. Este documento distingue con precisión lo **implementado y
probado** de lo que sigue **pendiente de ejecución** o de implementación.

## Estado real (29 de octubre de 2026)

| Bloque | Estado | Evidencia |
|--------|:------:|-----------|
| Estructura del proyecto (`uv`, Ruff, pre-commit, Makefile) | ✅ | `make check` en verde en CPU. |
| Núcleo matemático (Ω, energía de grafo, D, rango efectivo) | ✅ | `tests/test_e0.py`. |
| Proyección conjunta nuclear (sort + float64, tau=0, interior, idempotencia) | ✅ | `tests/test_e0.py`; oráculo CVXPY/CLARABEL. |
| Proyección espectral y por capas | ✅ | `tests/test_e0.py`. |
| Gradiente NUC por autograd | ✅ | `tests/test_e0.py`. |
| Spectral norm (pesos efectivos vs originales) | ✅ | `tests/test_e0.py`. |
| Reglas ADMM: fases, `tau0`, `rho`, doble reescalado | ✅ | `tests/test_e0.py`, `tests/test_training.py`. |
| PSGD-M (proyección en entrada de época y tras cada minibatch) | ✅ | `tests/test_training.py`. |
| Split determinista, prefijos piloto, normalización | ✅ | `tests/test_training.py`. |
| Manifiesto 36 piloto / 540 E2 + 60 E4 + 10 continuaciones | ✅ | `tests/test_training.py`. |
| Reanudación checkpoint ↔ run continuo | ✅ | `tests/test_training.py`. |
| E1: SC, QG sin unicidad, caso sharpness | ✅ (casos analíticos) | `tests/test_e1.py`. |
| E1: diagnóstico ADMM controlado con solución oráculo | ⏳ | Pendiente; requiere implementar el caso cuadrático completo. |
| Piloto de 36 runs | ⏳ | Código listo; **no ejecutado** (requiere datos/GPU). |
| E2–E4 (benchmark) | ⏳ | Runner listo; **no ejecutado**. |
| E3/E5/E6 (análisis, figuras, compresión) | 🟡 | Se exportan métricas y certificados; faltan figuras y compresión. |
| Evaluación final de test | 🔒 | Bloqueada por diseño (`evaluate-test` exige selección congelada). |
| Notebooks Kaggle | ✅ (generados) | `kaggle/`; no ejecutados en Kaggle desde esta máquina. |

**No hay resultados de benchmark en este repositorio.** Cualquier tabla o figura del
manuscrito sigue siendo provisional hasta ejecutar E0/E1, el piloto y el benchmark.

## Mapa especificación → módulos

| Sección de la especificación | Módulo | Funciones clave |
|------------------------------|--------|-----------------|
| §1 Objetos, Ω, A(W), E(A), D | `energy.py` | `omega`, `graph_adjacency`, `graph_energy`, `spectral_dimension` |
| §2 Proyección conjunta | `projection.py` | `project_joint`, `project_joint_torch`, `ProjectionPolicy` |
| §3 Operaciones espectrales | `projection.py`, `energy.py` | `project_spectral`, `spectral_summary`, `effective_rank` |
| §4 ADMM y schedule | `methods.py`, `train.py` | `budget_phase`, `adapt_rho`, `initial_tau0`, `train_run` |
| §5 PSGD-M y asignación por capas | `projection.py`, `train.py` | `allocate_radii`, `project_layerwise` |
| §6 Modelos y certificados | `evaluate.py` | `spectral_report`, `feasibility_report`, `evaluation_record` |
| §7 KKT / estabilidad | `evaluate.py`, `tests/test_e1.py` | cotas de perturbación y QG |
| §8 Interfaz pendiente | `train.py`, `runner.py`, `cli.py` | `train_run`, `execute_manifest` |
| Split y datos | `data.py` | `prepare_dataset`, `build_bundles`, `load_bundle` |
| Inventario y deduplicación | `manifest.py` | `generate_main_manifest`, `validate_manifest` |
| Ejecución con presupuesto | `runner.py` | `execute_manifest`, `RunOutcome` |

## Convenciones numéricas

- Entrenamiento en `float32`, sin AMP y con TF32 desactivado; `set_determinism` fija
  semillas y `torch.use_deterministic_algorithms(..., warn_only=True)`.
- Proyección en `float64` sobre CPU por defecto (`projection_device="cpu"`), y conversión
  de vuelta al dtype/dispositivo de entrenamiento. Este perfil se registra en cada run.
- Tolerancia primaria: `1e-6 + 1e-5 * max(1, |tau|)`. En E0 float64:
  `1e-10 + 1e-10 * max(1, tau)`.
- La proyección es una aproximación de punto flotante, **no** una cota rigurosa de
  redondeo. Si el resultado float32 no es factible, se reintenta una vez en float64 y se
  registra; si persiste, la run falla como `numerical_failure`.
- Nunca se usa SVD truncada en el suite principal.

## Decisiones de implementación que deben revisarse antes del benchmark

1. **`tau` para métodos sin presupuesto.** STD, WD, NUC y SN se entrenan una sola vez
   (no dependen de `alpha`) y se evalúan contra `Omega(W_STD)` (escala `alpha=1`). El
   análisis multi-`alpha` se hará reutilizando checkpoints, sin reentrenar.
2. **`tau0` de PSGD-M.** Se inicializa al final de la época 3 como
   `max(Omega(W^3), tau_final)`, igual que ADMM; así la transición 4–8 está bien definida.
3. **Proyección de PSGD-M en CPU.** El protocolo pide float64; la proyección por minibatch
   en GPU float64 sería lenta y menos portable. Se documenta como perfil numérico. Cambiarlo
   exige regenerar el manifiesto y repetir E0.
4. **Selección WD/NUC.** `analyze` produce una primera selección por media de
   `|Omega/tau - 1|`; la versión actual agrupa por seed y debe consolidarse por
   dataset/arquitectura/alpha antes de congelar tablas. Está marcado como pendiente.
5. **E4 continuación 25→100.** Copia el checkpoint padre y reanuda con `epochs=100`; exige
   que E2 esté completo. El runner falla con un mensaje claro si falta el padre.

## Qué falta para cerrar el trabajo

- Ejecutar `make e0` en CPU float64 y archivar `E0/E0_E1_report.json`.
- Ejecutar el piloto de 36 runs y medir tiempo/memoria por celda.
- Acordar el presupuesto y congelar `configs/main_manifest.jsonl`.
- Ejecutar E2 (540), E4 (60 + 10 continuaciones) en shards de Kaggle.
- Implementar E3/E5/E6: figuras, certificados por época, ratios E(A)/Ω, compresión real.
- Cerrar la selección por validación y **solo entonces** evaluar test una vez.
- Actualizar el manuscrito con resultados; no adelantar conclusiones.
