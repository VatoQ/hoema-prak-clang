"""FluxionCore: Pure Python API for numerical linear algebra."""

__version__ = "0.1.0"

from ._core import (
    DataType,
    NormType,
    FourierAlgorithm,
    FluxionError,
    DimensionError,
    MathError,
    BasicError,
    ensure_signatures,
)

from .matrix import Matrix

__all__ = [
    "DataType",
    "NormType",
    "FourierAlgorithm",
    "FluxionError",
    "DimensionError",
    "MathError",
    "BasicError",
    "ensure_signatures",
]
