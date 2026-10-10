"""DP Variant-driven synthetic event-log generation for Phase 6 interfaces."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from bul_project.behavior.models import Edge, Variant
from bul_project.privacy._validation import validate_experiment_seed
from bul_project.privacy.temporal_generator import TemporalDistributionResult


SYNTHETIC_START_TIMESTAMP = pd.Timestamp("2026-01-01 00:00:00")


def _variant_record(variant: Variant) -> list[str]:
    """Return variant record.

    Inputs:
        variant: Input value used by this operation.
    Outputs:
        A value of type `list[str]` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    return list(variant)


def _validate_postprocessed_count(value: float, name: str) -> int:
    """Validate postprocessed count.

    Inputs:
        value: Candidate value to validate or transform.
        name: Parameter or field name used in validation messages.
    Outputs:
        A value of type `int` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a non-negative integer DP count.")
    numeric = float(value)
    if (
        not math.isfinite(numeric)
        or numeric < 0
        or not numeric.is_integer()
    ):
        raise ValueError(f"{name} must be a non-negative integer DP count.")
    return int(numeric)


class MissingTemporalEdgeError(ValueError):
    """Raised when a sampled DP Variant contains an edge with no ``T'_e``."""

    def __init__(
        self,
        missing_edges: tuple[Edge, ...],
        metadata: dict[str, Any],
    ) -> None:
        """Initialize the instance and validate its construction inputs.

        Inputs:
            missing_edges: Input value used by this operation.
            metadata: Metadata associated with the artifact or experiment condition.
        Outputs:
            None.
        Paper role:
            Artifact provenance and representative external feasibility validation.
        """
        self.missing_edges = missing_edges
        self.metadata = metadata
        rendered = ", ".join(f"{source}->{target}" for source, target in missing_edges)
        super().__init__(f"Missing DP temporal distributions for edges: {rendered}")


@dataclass(frozen=True)
class SyntheticLogResult:
    """Canonical synthetic event table plus DP-generation audit metadata."""

    events: pd.DataFrame
    seed: int
    case_count: int
    metadata: dict[str, Any]

    def metadata_json(self) -> str:
        """Serialize generation metadata for later Phase 6 artifact writing."""

        return json.dumps(
            self.metadata,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


class SyntheticLogGenerator:
    """Generate a DP-consistent log from concrete DP Variants and ``T'_e``.

    ``OTHER`` has no concrete activity sequence and is accepted only as a
    separately recorded excluded count. Temporal input must be a
    ``TemporalDistributionResult`` so this interface cannot accept original
    per-edge time samples directly.
    """

    def __init__(self, *, seed: int) -> None:
        """Initialize the instance and validate its construction inputs.

        Inputs:
            seed: Random seed controlling reproducible execution.
        Outputs:
            None.
        Paper role:
            Artifact provenance and representative external feasibility validation.
        """
        self.seed = validate_experiment_seed(seed)

    def generate(
        self,
        variant_counts: Mapping[Variant, float],
        temporal_distributions: TemporalDistributionResult,
        *,
        other_count: float = 0,
    ) -> SyntheticLogResult:
        """Generate a canonical synthetic event log using only DP outputs.

        Input
        -----
        variant_counts:
            Post-processed DP counts for concrete Top-k activity-sequence
            tuples only. ``OTHER`` must not appear in this mapping.
        temporal_distributions:
            ``T'_e`` pools produced by ``TemporalDistributionGenerator`` from
            DP ``C'_E``, ``S'_T`` and ``Q'_T`` using the same experiment seed.
        other_count:
            DP ``OTHER`` count recorded in metadata and excluded from both
            normalization and ``N_case``.

        Output
        ------
        SyntheticLogResult
            A DataFrame with exactly ``case_id``, ``activity``, ``timestamp``
            and JSON-ready parameters/seed metadata. Concrete Variant
            probabilities are recomputed as ``C'_v/sum(C'_v)`` and
            ``N_case=sum(C'_v)`` excludes ``OTHER``.

        Paper experiment step
        ---------------------
        Phase 6 Synthetic Log interface: sample concrete DP Variants, traverse
        their edge order, sample intervals with replacement from the matching
        DP-generated ``T'_e`` pool, and accumulate timestamps forward from
        ``2026-01-01 00:00:00``. Missing temporal edges fail explicitly; no
        original trace, Variant distribution, edge time, or fallback value is
        accessed.
        """

        if not isinstance(temporal_distributions, TemporalDistributionResult):
            raise TypeError(
                "temporal_distributions must be a DP TemporalDistributionResult."
            )
        if temporal_distributions.seed != self.seed:
            raise ValueError(
                "Temporal and Synthetic Log generators must use the same experiment seed."
            )
        excluded_other_count = _validate_postprocessed_count(
            other_count, "other_count"
        )
        if not variant_counts:
            raise ValueError("variant_counts must contain concrete DP Variants.")

        checked_counts: dict[Variant, int] = {}
        for variant, count in variant_counts.items():
            if (
                not isinstance(variant, tuple)
                or not variant
                or any(
                    not isinstance(activity, str) or not activity
                    for activity in variant
                )
            ):
                raise ValueError(
                    "variant_counts keys must be non-empty concrete activity tuples; "
                    "the OTHER label is not a synthetic Variant."
                )
            checked_counts[variant] = _validate_postprocessed_count(
                count, f"variant_counts[{variant!r}]"
            )

        case_count = sum(checked_counts.values())
        if case_count <= 0:
            raise ValueError(
                "Concrete DP Variant counts must have positive total mass; "
                "OTHER is excluded from N_case."
            )
        concrete_variants = tuple(
            variant for variant in sorted(checked_counts) if checked_counts[variant] > 0
        )
        probabilities = np.asarray(
            [checked_counts[variant] / case_count for variant in concrete_variants],
            dtype=float,
        )

        rng = np.random.default_rng(self.seed)
        sampled_indices = rng.choice(
            len(concrete_variants),
            size=case_count,
            replace=True,
            p=probabilities,
        )
        sampled_variants = tuple(
            concrete_variants[int(index)] for index in sampled_indices
        )
        sampled_variant_counts = Counter(sampled_variants)
        required_edges = {
            edge
            for variant in sampled_variants
            for edge in zip(variant, variant[1:])
        }
        available_edges = set(temporal_distributions.samples)
        missing_edges = tuple(sorted(required_edges - available_edges))
        base_metadata: dict[str, Any] = {
            "module": "SyntheticLogGenerator",
            "dp_post_processing_only": True,
            "raw_data_access": False,
            "seed": self.seed,
            "start_timestamp": SYNTHETIC_START_TIMESTAMP.isoformat(),
            "timestamp_rule": "timestamp[i+1] = timestamp[i] + T'_e",
            "interval_sampling": "with_replacement_from_DP_T'_e",
            "other_policy": "excluded_from_synthetic_generation",
            "excluded_other_count": excluded_other_count,
            "case_count_rule": "sum(concrete C'_V)",
            "case_count": case_count,
            "concrete_variant_distribution": [
                {
                    "activities": _variant_record(variant),
                    "dp_count": checked_counts[variant],
                    "synthetic_probability": checked_counts[variant] / case_count,
                }
                for variant in concrete_variants
            ],
            "sampled_variant_counts": [
                {
                    "activities": _variant_record(variant),
                    "sampled_count": sampled_variant_counts[variant],
                }
                for variant in sorted(sampled_variant_counts)
            ],
            "temporal_generator_seed": temporal_distributions.seed,
        }
        if missing_edges:
            failure_metadata = {
                **base_metadata,
                "status": "failed_missing_temporal_edges",
                "missing_edges": [
                    {"source": source, "target": target}
                    for source, target in missing_edges
                ],
            }
            raise MissingTemporalEdgeError(missing_edges, failure_metadata)

        records: list[dict[str, Any]] = []
        for case_number, variant in enumerate(sampled_variants, start=1):
            case_id = f"synthetic_{case_number:06d}"
            timestamp = SYNTHETIC_START_TIMESTAMP
            records.append(
                {
                    "case_id": case_id,
                    "activity": variant[0],
                    "timestamp": timestamp,
                }
            )
            for edge, activity in zip(zip(variant, variant[1:]), variant[1:]):
                pool = temporal_distributions.samples[edge]
                if not pool:
                    # TemporalDistributionGenerator guarantees non-empty pools;
                    # retain a strict guard against hand-built invalid objects.
                    raise RuntimeError(f"DP temporal sample pool is empty for edge {edge!r}.")
                interval = float(rng.choice(pool))
                if not math.isfinite(interval) or interval < 0:
                    raise RuntimeError(
                        f"DP temporal sample is invalid for edge {edge!r}: {interval}"
                    )
                timestamp = timestamp + pd.to_timedelta(interval, unit="s")
                records.append(
                    {
                        "case_id": case_id,
                        "activity": activity,
                        "timestamp": timestamp,
                    }
                )

        events = pd.DataFrame.from_records(
            records,
            columns=("case_id", "activity", "timestamp"),
        )
        metadata = {
            **base_metadata,
            "status": "success",
            "event_count": len(events),
            "missing_edges": [],
        }
        return SyntheticLogResult(
            events=events,
            seed=self.seed,
            case_count=case_count,
            metadata=metadata,
        )
