# =============================================================================
# RAM Graph Energy — automatización de tareas
# =============================================================================

SHELL := /bin/bash

.DEFAULT_GOAL := help
.PHONY: help install dev pre-commit-install pre-commit lint format format-check typecheck test e0 check build notebooks clean

help: ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Sincroniza el entorno virtual e instala dependencias
	uv sync

dev: install pre-commit-install ## Prepara el entorno de desarrollo (sync + hooks)

pre-commit-install: ## Instala los hooks de pre-commit en el repositorio local
	uv run pre-commit install

pre-commit: ## Ejecuta todos los hooks de pre-commit sobre el repositorio
	uv run pre-commit run --all-files

lint: ## Ejecuta Ruff (lint)
	uv run ruff check .

format: ## Formatea el código con Ruff
	uv run ruff format .

format-check: ## Verifica el formato sin modificarlo
	uv run ruff format --check .

typecheck: ## (Omitido) El proyecto no usa ty; se retiró para mantener make check en verde
	@echo "ty no está configurado en este proyecto (ver CONSTITUTION.md §3)."

test: ## Ejecuta las pruebas
	uv run pytest

e0: ## Ejecuta las verificaciones E0/E1 y escribe E0/E0_E1_report.json
	uv run python -m ram_experiments.cli verify --suite E0,E1 --out E0

check: lint format-check test ## Verificación completa (lint + formato + pruebas)

build: ## Construye el paquete (sdist + wheel)
	uv build

notebooks: ## Instala el kernel de Jupyter del proyecto (grupo 'notebooks')
	uv sync --group notebooks
	uv run python -m ipykernel install --user --name energia-2 --display-name "RAM Graph Energy"

clean: ## Limpia cachés y artefactos
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
