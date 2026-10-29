"""Reference implementation of protocol RAM-2026-10-13-v1.

The package implements the mathematical objects, the optimization methods and the
run bookkeeping described in ``MATHEMATICAL_IMPLEMENTATION_SPECIFICATION.md`` and
``EXPERIMENTAL_PROTOCOL.md``. It does not ship datasets, trained checkpoints or
benchmark results.
"""

from __future__ import annotations

from ram_experiments.config import (
    ALPHAS,
    ARCHITECTURES,
    METHODS,
    PROTOCOL_VERSION,
    SEEDS,
    RunConfig,
    TrainConfig,
)

__all__ = [
    "ALPHAS",
    "ARCHITECTURES",
    "METHODS",
    "PROTOCOL_VERSION",
    "SEEDS",
    "RunConfig",
    "TrainConfig",
]

__version__ = "0.1.0"
