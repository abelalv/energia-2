"""RAM Graph Energy — prueba de humo."""

from __future__ import annotations

import ram_experiments


def test_el_paquete_expone_una_version() -> None:
    # Arrange / Act
    version = getattr(ram_experiments, "__version__", None)

    # Assert
    assert version is not None
    assert isinstance(version, str)
