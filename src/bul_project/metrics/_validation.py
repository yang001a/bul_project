"""Shared input alignment and validation for established Phase 4 metrics."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import TypeAlias

import numpy as np


VectorInput: TypeAlias = Mapping[Hashable, float] | Sequence[float] | np.ndarray
SampleInput: TypeAlias = Sequence[float] | np.ndarray


def _as_numeric_vector(values: object, name: str) -> np.ndarray:
    """Convert input values to a validated numeric vector.

    Inputs:
        values: Values processed by the operation.
        name: Parameter or field name used in validation messages.
    Outputs:
        A value of type `np.ndarray` produced by the operation.
    Paper role:
        Behavioral utility loss component evaluation.
    """
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a numeric one-dimensional vector.")
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric one-dimensional vector.") from exc
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(vector < 0):
        raise ValueError(f"{name} must contain only non-negative values.")
    return vector


def align_vectors(
    reference: VectorInput,
    comparison: VectorInput,
) -> tuple[np.ndarray, np.ndarray]:
    """Align mappings on their union or validate equal-length vectors."""

    reference_is_mapping = isinstance(reference, Mapping)
    comparison_is_mapping = isinstance(comparison, Mapping)
    if reference_is_mapping != comparison_is_mapping:
        raise TypeError("reference and comparison must both be mappings or both vectors.")

    if reference_is_mapping and comparison_is_mapping:
        reference_mapping = reference
        comparison_mapping = comparison
        keys = sorted(
            set(reference_mapping) | set(comparison_mapping),
            key=lambda key: (type(key).__name__, repr(key)),
        )
        if not keys:
            raise ValueError("Metric distributions must contain at least one coordinate.")
        reference_values = [reference_mapping.get(key, 0.0) for key in keys]
        comparison_values = [comparison_mapping.get(key, 0.0) for key in keys]
        left = _as_numeric_vector(reference_values, "reference")
        right = _as_numeric_vector(comparison_values, "comparison")
    else:
        left = _as_numeric_vector(reference, "reference")
        right = _as_numeric_vector(comparison, "comparison")

    if left.shape != right.shape:
        raise ValueError("reference and comparison vectors must have equal length.")
    return left, right


def normalize_distribution(vector: np.ndarray, name: str) -> np.ndarray:
    """Normalize non-negative mass without changing its coordinate space."""

    total = float(np.sum(vector))
    if total <= 0:
        raise ValueError(f"{name} distribution must have positive total mass.")
    return vector / total


def validate_samples(samples: SampleInput, name: str) -> np.ndarray:
    """Validate one non-empty transition-time sample vector."""

    return _as_numeric_vector(samples, name)
