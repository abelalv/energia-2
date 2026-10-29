# CONSTITUTION — RAM Graph Energy

> **Verdad central del proyecto.** Este documento define el propósito, el stack, la
> estructura, las convenciones y el flujo de trabajo de `RAM Graph Energy`. Si algo aquí
> contradice al código, el código manda **y este archivo se actualiza**.

> ⚠️ **Directiva de mantenimiento:** cada vez que se realicen cambios significativos
> (nuevas funcionalidades, cambios arquitectónicos, adición de clases/módulos o ajustes
> de tecnologías), **se debe actualizar este archivo en el mismo commit** para mantener
> la verdad central del proyecto bien documentada.

---

## 0. Identidad del proyecto

| Campo | Valor |
|-------|-------|
| **Nombre** | RAM Graph Energy |
| **Descripción** | Implementación y benchmark del presupuesto global de norma nuclear en MLPs (protocolo RAM-2026-10-13-v1), ejecutable en Kaggle. |
| **Repositorio** | https://github.com/abelalv/energia-2 |
| **Autoría** | Abel Álvarez Bustos — aalvarezb@unal.edu.co |
| **Licencia** | MIT |
| **Rama principal** | `main` |
| **Estado** | Implementación E0/E1 completa; benchmark pendiente de ejecución 🚧 |

## 1. Propósito y no-propósito

### Propósito

Implementar de forma auditable el protocolo `RAM-2026-10-13-v1` para comparar STD, WD,
NUC, SN, ADMM conjunto y PSGD-M bajo un presupuesto global de norma nuclear
`C_tau = {W : 2 Σ‖W_l‖_* ≤ tau}` en MLPs sobre MNIST y Fashion-MNIST, y ejecutarlo en
Kaggle con presupuesto de sesión, reanudación y trazabilidad completa.

### No-propósito

No es una librería de propósito general, no mide consumo eléctrico, FLOPs ni compresión
real de despliegue, y no afirma superioridad predictiva de ADMM. No sustituye las
demostraciones del manuscrito ni autoriza conclusiones asintóticas a partir de residuales
finitos. Tampoco es el notebook histórico: ese material se conserva aparte, sin ejecutar.

## 2. Alcance

### Dentro del alcance

- Núcleo matemático: Ω(W), A(W), E(A), D, proyección nuclear conjunta, proyección
  espectral y por capas, rango efectivo.
- Modelos 3L/6L/9L, datasets MNIST/Fashion-MNIST, split determinista y normalización.
- Métodos STD, WD, NUC, SN, ADMM y PSGD-M con el schedule 3/5/17.
- Manifiesto de 540 runs E2, 60 E4 y 10 continuaciones; piloto de 36 runs.
- Verificaciones E0/E1, certificados de factibilidad y diagnóstico de estacionariedad
  cuando sea aplicable.
- Ejecución en Kaggle con shards, `--max-seconds` y reanudación por checkpoint.

### Fuera del alcance (por ahora)

- Ejecutar el benchmark sin presupuesto medido y acordado.
- Evaluar el test antes de congelar la selección por validación.
- Compresión real (E6) sin artefacto y medición; solo se reportan propiedades espectrales.
- Cambiar fórmulas, tolerancias o el grid WD/NUC después de ver resultados.
- Reparar o ejecutar el notebook histórico.

## 3. Stack técnico

| Categoría | Tecnología |
|-----------|------------|
| **Lenguaje** | Python 3.11+ |
| **Gestor de paquetes y entorno** | [`uv`](https://docs.astral.sh/uv/) |
| **Build backend** | `uv_build` (src-layout, módulo `ram_experiments`) |
| **Paquete Python** | `ram_experiments` |
| **Cómputo** | PyTorch 2.x + torchvision (CPU local, GPU en Kaggle) |
| **Oráculo E0** | CVXPY + CLARABEL (grupo `dev`/extra `oracle`) |
| **Testing** | [`pytest`](https://docs.pytest.org/) |
| **Linting / Formato** | [Ruff](https://docs.astral.sh/ruff/) |
| **Pre-commit** | [pre-commit](https://pre-commit.com) |
| **Notebooks** | Jupyter (`ipykernel`, grupo `notebooks`) + [`nbstripout`](https://github.com/kynan/nbstripout) |
| **Ejecución remota** | Kaggle Notebooks (GPU T4 x2 / P100, 12 h/sesión) |

> ⚠️ **Regla dura:** toda gestión de dependencias y entornos pasa por `uv`. Está
> **prohibido** `pip` y `requirements.txt`. Única excepción documentada: dentro de
> Kaggle/Colab se instala solo el paquete con `pip install -e . --no-deps` (el entorno no
> es nuestro y trae su propio `torch`/CUDA).

## 4. Estructura del repositorio

```text
energia-2/
├── src/ram_experiments/     # paquete (src-layout)
│   ├── config.py            # constantes, RunConfig, hash canónico
│   ├── energy.py            # Ω, A(W), E(A), D, rango efectivo
│   ├── projection.py        # proyección conjunta/espectral/por capas
│   ├── models.py            # MLP 3L/6L/9L, SN, pesos efectivos
│   ├── data.py              # splits, normalización, .npz
│   ├── methods.py           # objetivos, schedule, adaptación de rho
│   ├── train.py             # bucle, checkpoints, reanudación
│   ├── evaluate.py          # métricas, espectros, certificados
│   ├── manifest.py          # inventario y deduplicación
│   ├── runner.py            # ejecución con presupuesto de sesión
│   └── cli.py               # interfaz `ram-experiments`
├── tests/                   # E0, E1, datos, manifiesto, smoke de entrenamiento
├── configs/                 # snapshot del protocolo y manifiestos
├── kaggle/                  # notebooks + kernel-metadata.json
├── scripts/                 # generadores de notebooks y datasets
├── docs/                    # implementación, Kaggle, ADRs
├── notebooks/               # exploración (no fuente de resultados)
├── data/                    # raw/processed fuera de Git
├── figures/                 # figuras generadas por código
├── runs/                    # artefactos de ejecución (ignorado por Git)
├── mathematical_checks/     # 227 checks originales preservados
├── manuscript/              # manuscrito revisado
├── historical_reference/    # material histórico para auditoría
├── CONSTITUTION.md          # este archivo — verdad central
├── README.md                # puerta de entrada pública
├── Makefile                 # automatización de tareas
├── pyproject.toml           # metadatos y dependencias (uv)
├── uv.lock                  # lockfile determinista (versionado)
└── .python-version          # versión de Python fijada
```

## 5. Convenciones de código

- **Estilo:** [PEP 8](https://peps.python.org/pep-0008/) y [PEP 257](https://peps.python.org/pep-0257/),
  impuesto por **Ruff** (`line-length = 100`, comillas dobles).
- **Tipado:** anotaciones en funciones y firmas públicas.
- **Nomenclatura:** `snake_case` (funciones/variables), `PascalCase` (clases),
  `UPPER_CASE` (constantes).
- **Docstrings:** en módulos, clases y funciones públicas.
- **Pruebas:** `pytest` con el patrón **AAA** (*Arrange · Act · Assert*).
- **Idioma:** UI, comentarios y commits en español; identificadores de código en inglés.
- **Reglas propias (invariantes del dominio):**
  - `Omega(W) = 2 * sum_l ||W_l||_*` y `E(A)` son cantidades distintas; nunca se
    intercambian en tablas ni figuras.
  - La matriz de adyacencia es simétrica, con signos, bloques consecutivos y sin biases;
    no se toma valor absoluto entrada a entrada.
  - El presupuesto es `C_tau = {W : 2 * sum_l ||W_l||_* <= tau}`; `tau = 2` implica suma
    nuclear `<= 1`.
  - La proyección es pura: no muta la entrada y no presupuesta los biases.
  - ADMM no sobrescribe `W <- Z`; el modelo primario es `W` del último epoch completo.
  - PSGD-M conserva el buffer de momentum y proyecta en la entrada de época y tras cada
    minibatch; no proyecta el buffer.
  - NUC conserva autograd; `detach`, NumPy o float en la penalización rompen el objetivo.
  - La selección se hace por validación; el test se evalúa una sola vez tras congelar.
  - Los fallos se registran con su estado; no se sustituyen por ceros ni se descartan por
    accuracy.
- **Reproducibilidad (investigación):**
  - Semillas fijadas y registradas en cada experimento: 42, 123, 456, 789, 1024.
  - Split determinista con `PCG64(20261013)`; índices y hashes publicados.
  - Los notebooks importan desde el paquete; el código reutilizable vive en `src/`.
  - Toda figura de `figures/` se regenera con un script; nunca se edita a mano.
  - Fuentes de datos documentadas en `data/README.md`.
  - Cada run guarda `config.json`, `environment.json`, métricas, espectros, checkpoint y
    hashes; `run_id` y `config_hash` identifican la configuración científica.
  - La proyección float64 en CPU es un **perfil numérico declarado**; cambiarlo exige un
    hash nuevo y repetir E0.


## 6. Calidad y automatización

| Herramienta | Rol | Comando |
|-------------|-----|---------|
| `uv` | Entornos y dependencias | `uv sync` |
| Ruff | Lint + formato | `make lint` / `make format` |
| `pytest` | Pruebas | `make test` |
| `pre-commit` | Hooks antes de cada commit | `make pre-commit` |
| `nbstripout` | Limpia salidas de notebooks | automático (pre-commit) |
| E0/E1 | Puerta matemática | `make e0` |

- Ningún commit se salta los hooks (`--no-verify` está prohibido salvo causa justificada y documentada).
- `make check` debe estar en verde antes de cada push (no hay CI configurada).
- Toda dependencia nueva entra con `uv add` / `uv add --dev` (nunca editando `uv.lock` a mano).
- `ty` se retiró del stack: los stubs de PyTorch generaban falsos positivos y no aportaban
  una garantía adicional a E0/E1. Si se reintroduce, debe pasar `make check` sin exclusiones
  globales.

## 7. Repositorio y Git

- **Remoto (SSH):** `git@github.com:abelalv/energia-2.git`
- **Rama principal:** `main`
- **Estrategia de ramas:** `feat/...`, `fix/...`, `docs/...`, `refactor/...`, `test/...`, `chore/...`; integración vía Pull Request.
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/) —
  `feat:`, `fix:`, `docs:`, `refactor:`, `style:`, `test:`, `chore:`.
- **Identidad git:** se resuelve sola dentro de la carpeta de repositorios configurada con
  `includeIf` (ver §11).

## 8. Flujo diario (commit + push)

```bash
# 1. Ver cambios
git status

# 2. Verificación completa (lint + formato + pruebas)
make check

# 3. Agregar y commitear (los hooks se ejecutan solos)
git add -A
git commit -m "tipo: mensaje descriptivo"

# 4. Subir a GitHub
git push
```

## 9. Historial de decisiones técnicas

| Fecha | Decisión | Motivo |
|-------|----------|--------|
| 2026-10-29 | Adoptar `uv` como gestor único de entornos y dependencias | Reproducibilidad; prohibido `pip`/`requirements.txt` |
| 2026-10-29 | Adoptar Ruff (lint + formato) y pre-commit | Calidad automatizada antes de cada commit |
| 2026-10-29 | Adoptar Conventional Commits | Historial legible y automatizable |
| 2026-10-29 | Adoptar licencia MIT y `CONSTITUTION.md` como fuente única de verdad | Claridad legal y operativa |
| 2026-10-29 | Empaquetar el código como `ram_experiments` (no `energia_2`) | Coincide con la interfaz propuesta en `EXPERIMENTAL_PROTOCOL.md` |
| 2026-10-29 | Python 3.11 como versión objetivo | Compatibilidad con PyTorch y con la imagen de Kaggle; `>=3.11` en `pyproject.toml` |
| 2026-10-29 | Proyección nuclear en CPU/float64; entrenamiento en GPU/float32 | Estabilidad numérica y coste predecible; perfil declarado en el hash |
| 2026-10-29 | Retirar `ty` del stack | Stubs de PyTorch con falsos positivos; E0/E1 cubren la corrección numérica |
| 2026-10-29 | Kaggle como plataforma de ejecución principal | GPU gratuita/cuota semanal, notebooks reproducibles y datasets privados |
| 2026-10-29 | `tau0` de PSGD-M = `max(Omega(W^3), tau_final)` | Define la transición 4–8 sin ambigüedad, igual que ADMM |
| 2026-10-29 | Manifiesto: 540 E2 + 60 E4 + 10 continuaciones | Refleja el inventario del protocolo sin duplicar WD/NUC/SN por alpha |

## 10. Notas y tareas pendientes

- [ ] Configurar el remoto `origin` y proteger `main` en GitHub.
- [ ] Ejecutar `make e0` y archivar `E0/E0_E1_report.json`.
- [ ] Ejecutar el piloto de 36 runs y medir tiempo/memoria por celda.
- [ ] Implementar el caso E1 de ADMM controlado con solución oráculo.
- [ ] Consolidar la selección WD/NUC por dataset/arquitectura/alpha (no por seed).
- [ ] Implementar E3/E5/E6: figuras, certificados por época, ratios E(A)/Ω y compresión.
- [ ] Congelar `configs/main_manifest.jsonl` tras el presupuesto acordado.
- [ ] Ejecutar E2/E4 en shards de Kaggle y exportar el paquete de resultados.
- [ ] Evaluar test una sola vez, después de cerrar la selección por validación.
- [ ] Actualizar el manuscrito con resultados; no adelantar conclusiones.

## 11. Configuración local (no trackear)

> Configuración específica de la máquina local. **NO incluir en commits.**

Dentro de la carpeta de proyectos, `~/.gitconfig` aplica:

```ini
[includeIf "gitdir:~/Repos/"]
    path = ~/.gitconfig-personal
```

`~/.gitconfig-personal`:

```ini
[user]
    name = <Tu Nombre>
    email = <tu-correo>
[core]
    sshCommand = ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes
```

> Ajusta la carpeta y tus datos. Guía completa (macOS/Windows) en el skill
> `proyecto-abel` → `references/git-identity.md`.
