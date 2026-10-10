"""DP-only reconstruction of per-edge temporal sample distributions."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
from scipy.stats import truncnorm

from bul_project.behavior.models import Edge
from bul_project.config import DP_DENOMINATOR_DELTA

from ._validation import (
    validate_experiment_seed,
    validate_t_max,
)


def _validate_dp_value(
    value: float, name: str, *, non_negative: bool
) -> float:
    """Validate dp value.

    Inputs:
        value: Candidate value to validate or transform.
        name: Parameter or field name used in validation messages.
        non_negative: Input value used by this operation.
    Outputs:
        A value of type `float` produced by the operation.
    Paper role:
        Case-level differential-privacy release and temporal post-processing.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite numeric DP statistic.")
    numeric = float(value)
    if not math.isfinite(numeric) or (non_negative and numeric < 0):
        qualifier = " non-negative" if non_negative else ""
        raise ValueError(f"{name} must be a finite{qualifier} numeric DP statistic.")
    return numeric


def _edge_record(edge: Edge) -> dict[str, str]:
    """Return edge record.

    Inputs:
        edge: Input value used by this operation.
    Outputs:
        A value of type `dict[str, str]` produced by the operation.
    Paper role:
        Case-level differential-privacy release and temporal post-processing.
    """
    return {"source": edge[0], "target": edge[1]}


@dataclass(frozen=True)
class TemporalDistributionResult:
    """Generated ``T'_e`` samples and their complete DP-only audit metadata."""

    samples: dict[Edge, tuple[float, ...]]
    means: dict[Edge, float]
    variances: dict[Edge, float]
    standard_deviations: dict[Edge, float]
    sample_counts: dict[Edge, int]
    seed: int
    t_max: float
    delta: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible result and metadata payload."""

        edges = []
        for edge in sorted(self.samples):
            edges.append(
                {
                    **_edge_record(edge),
                    "mean_seconds": self.means[edge],
                    "variance_seconds_squared": self.variances[edge],
                    "standard_deviation_seconds": self.standard_deviations[edge],
                    "sample_count": self.sample_counts[edge],
                    "samples_seconds": list(self.samples[edge]),
                }
            )
        return {
            "name": "DP_temporal_distributions",
            "seed": self.seed,
            "t_max": self.t_max,
            "delta": self.delta,
            "edges": edges,
            "metadata": self.metadata,
        }

    def metadata_json(self) -> str:
        """Serialize generation metadata for later Phase 6 artifact writing."""

        return json.dumps(
            self.metadata,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


class TemporalDistributionGenerator:
    """Generate bounded ``T'_e`` using only DP edge/time statistics.

    The generator has no trace, raw-log, or raw-time input. It reconstructs
    moments from ``C'_E``, ``S'_T`` and ``Q'_T`` and samples a truncated normal
    distribution on ``[0,Tmax]`` with one recorded experiment seed.
    """

    def __init__(
        self,
        *,
        t_max: float,
        seed: int,
        delta: float = DP_DENOMINATOR_DELTA,
    ) -> None:
        """Initialize the instance and validate its construction inputs.

        Inputs:
            t_max: Input value used by this operation.
            seed: Random seed controlling reproducible execution.
            delta: Input value used by this operation.
        Outputs:
            None.
        Paper role:
            Case-level differential-privacy release and temporal post-processing.
        """
        self.t_max = validate_t_max(t_max)
        self.seed = validate_experiment_seed(seed)
        self.delta = _validate_dp_value(delta, "delta", non_negative=True)
        if self.delta == 0:
            raise ValueError("delta must be strictly positive.")

    def generate(
        self,
        edge_counts: Mapping[Edge, float],
        time_sums: Mapping[Edge, float],
        time_squared_sums: Mapping[Edge, float],
    ) -> TemporalDistributionResult:
        """Generate DP-consistent per-edge temporal samples.

        Input
        -----
        edge_counts:
            DP post-processed ``C'_E(e)`` values.
        time_sums:
            DP post-processed first moments ``S'_T(e)=sum(t)``.
        time_squared_sums:
            DP second moments ``Q'_T(e)=sum(t^2)``. A noisy negative value is
            permitted; the specified variance floor handles it without raw
            sample correction.

        Output
        ------
        TemporalDistributionResult
            Generated ``T'_e`` pools, reconstructed moments and JSON-ready
            metadata including DP inputs, bounds, formulas and seed.

        Paper experiment step
        ---------------------
        Phase 6 Temporal Reconstruction: calculate
        ``mu'=S'/max(C',delta)`` and
        ``variance'=max(Q'/max(C',delta)-mu'^2,delta)``, then draw
        ``max(round(C'),1)`` samples from ``TruncatedNormal(mu',variance')``
        bounded to ``[0,Tmax]``. This is DP post-processing and never accesses
        original traces or original time samples.
        """

        domains = (set(edge_counts), set(time_sums), set(time_squared_sums))
        if not domains[0]:
            raise ValueError("DP temporal statistics must contain at least one edge.")
        if domains[0] != domains[1] or domains[0] != domains[2]:
            raise ValueError(
                "edge_counts, time_sums and time_squared_sums must use exactly "
                "the same frozen DP edge domain."
            )
        for edge in domains[0]:
            if (
                not isinstance(edge, tuple)
                or len(edge) != 2
                or any(not isinstance(label, str) or not label for label in edge)
            ):
                raise ValueError("Every temporal-statistics key must be a valid Edge.")

        rng = np.random.default_rng(self.seed)
        samples: dict[Edge, tuple[float, ...]] = {}
        means: dict[Edge, float] = {}
        variances: dict[Edge, float] = {}
        standard_deviations: dict[Edge, float] = {}
        sample_counts: dict[Edge, int] = {}
        edge_metadata: list[dict[str, Any]] = []

        for edge in sorted(domains[0]):
            count = _validate_dp_value(
                edge_counts[edge], f"edge_counts[{edge!r}]", non_negative=True
            )
            first_sum = _validate_dp_value(
                time_sums[edge], f"time_sums[{edge!r}]", non_negative=True
            )
            second_sum = _validate_dp_value(
                time_squared_sums[edge],
                f"time_squared_sums[{edge!r}]",
                non_negative=False,
            )
            denominator = max(count, self.delta)
            mean = first_sum / denominator
            raw_variance = second_sum / denominator - mean**2
            variance = max(raw_variance, self.delta)
            standard_deviation = math.sqrt(variance)
            generated_count = max(int(math.floor(count + 0.5)), 1)

            if self.t_max == 0:
                generated = np.zeros(generated_count, dtype=float)
            else:
                lower = (0.0 - mean) / standard_deviation
                upper = (self.t_max - mean) / standard_deviation
                generated = np.asarray(
                    truncnorm.rvs(
                        lower,
                        upper,
                        loc=mean,
                        scale=standard_deviation,
                        size=generated_count,
                        random_state=rng,
                    ),
                    dtype=float,
                )
            if generated.ndim != 1 or generated.size != generated_count:
                raise RuntimeError("Truncated-normal generator returned an invalid shape.")
            if not np.all(np.isfinite(generated)):
                raise RuntimeError("Truncated-normal generator returned non-finite samples.")
            generated = np.clip(generated, 0.0, self.t_max)

            samples[edge] = tuple(float(value) for value in generated)
            means[edge] = float(mean)
            variances[edge] = float(variance)
            standard_deviations[edge] = float(standard_deviation)
            sample_counts[edge] = generated_count
            edge_metadata.append(
                {
                    **_edge_record(edge),
                    "dp_edge_count": count,
                    "dp_time_sum": first_sum,
                    "dp_time_squared_sum": second_sum,
                    "denominator": denominator,
                    "reconstructed_mean_seconds": mean,
                    "raw_variance_seconds_squared": raw_variance,
                    "reconstructed_variance_seconds_squared": variance,
                    "generated_sample_count": generated_count,
                }
            )

        metadata: dict[str, Any] = {
            "module": "TemporalDistributionGenerator",
            "status": "success",
            "dp_post_processing_only": True,
            "raw_data_access": False,
            "seed": self.seed,
            "distribution": "truncated_normal",
            "bounds_seconds": [0.0, self.t_max],
            "delta": self.delta,
            "sample_count_rule": "max(round(C'_E(e)), 1)",
            "mean_rule": "S'_T(e) / max(C'_E(e), delta)",
            "variance_rule": "max(Q'_T(e) / max(C'_E(e), delta) - mu'_e^2, delta)",
            "edges": edge_metadata,
        }
        return TemporalDistributionResult(
            samples=samples,
            means=means,
            variances=variances,
            standard_deviations=standard_deviations,
            sample_counts=sample_counts,
            seed=self.seed,
            t_max=self.t_max,
            delta=self.delta,
            metadata=metadata,
        )
