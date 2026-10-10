"""DP-only bridge from authenticated checkpoints to a synthetic event log.

This module deliberately accepts an in-memory ``experiment_metadata`` mapping,
not a raw trace, event log, evaluation log, or filesystem path.  Authentication
of the checkpoint bytes is a caller responsibility; this bridge validates the
scientific identity and strictly deserializes only the DP release fields needed
by the existing Phase 6 generators.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

from bul_project.behavior.models import Edge, Variant
from bul_project.validation.synthetic_log import (
    SyntheticLogGenerator,
    SyntheticLogResult,
)
from bul_project.privacy.temporal_generator import (
    TemporalDistributionGenerator,
    TemporalDistributionResult,
)


_DFG_FIELDS = frozenset({"source", "target", "count", "probability"})
_TEMPORAL_FIELDS = frozenset(
    {
        "source",
        "target",
        "mean_seconds",
        "sum_seconds",
        "squared_sum_seconds2",
    }
)
_VARIANT_FIELDS = frozenset({"activities", "count", "probability"})
_OTHER_FIELDS = frozenset({"label", "count", "probability"})
_TEMPORAL_SCIENCE_FIELDS = (
    "module",
    "status",
    "dp_post_processing_only",
    "raw_data_access",
    "seed",
    "distribution",
    "bounds_seconds",
    "delta",
    "sample_count_rule",
    "mean_rule",
    "variance_rule",
)
_TEMPORAL_EDGE_SCIENCE_FIELDS = (
    "dp_edge_count",
    "dp_time_sum",
    "dp_time_squared_sum",
    "denominator",
    "reconstructed_mean_seconds",
    "raw_variance_seconds_squared",
    "reconstructed_variance_seconds_squared",
    "generated_sample_count",
)


class ProcessQualityReleaseError(ValueError):
    """Raised when authenticated metadata is scientifically inconsistent."""


@dataclass(frozen=True)
class ProcessQualityReleaseResult:
    """Strictly parsed DP inputs and the two existing Phase 6 outputs."""

    edge_counts: dict[Edge, int]
    time_sums: dict[Edge, float]
    time_squared_sums: dict[Edge, float]
    variant_counts: dict[Variant, int]
    other_count: int
    temporal_distributions: TemporalDistributionResult
    synthetic_log: SyntheticLogResult
    metadata: dict[str, Any]


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Return mapping.

    Inputs:
        value: Candidate value to validate or transform.
        name: Parameter or field name used in validation messages.
    Outputs:
        A value of type `Mapping[str, Any]` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    if not isinstance(value, Mapping):
        raise ProcessQualityReleaseError(f"{name} must be a mapping.")
    return value


def _records(value: Any, name: str) -> list[Any]:
    """Return records.

    Inputs:
        value: Candidate value to validate or transform.
        name: Parameter or field name used in validation messages.
    Outputs:
        A value of type `list[Any]` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    if not isinstance(value, list):
        raise ProcessQualityReleaseError(f"{name} must be a JSON list.")
    if not value:
        raise ProcessQualityReleaseError(f"{name} must not be empty.")
    return value


def _require(mapping: Mapping[str, Any], key: str, name: str) -> Any:
    """Return require.

    Inputs:
        mapping: Input value used by this operation.
        key: Input value used by this operation.
        name: Parameter or field name used in validation messages.
    Outputs:
        A value of type `Any` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    if key not in mapping:
        raise ProcessQualityReleaseError(f"{name} is missing required field {key!r}.")
    return mapping[key]


def _exact_fields(
    mapping: Mapping[str, Any], expected: frozenset[str], name: str
) -> None:
    """Return exact fields.

    Inputs:
        mapping: Input value used by this operation.
        expected: Input value used by this operation.
        name: Parameter or field name used in validation messages.
    Outputs:
        None.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    actual = frozenset(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ProcessQualityReleaseError(
            f"{name} fields differ from the formal checkpoint schema; "
            f"missing={missing}, unexpected={unexpected}."
        )


def _label(value: Any, name: str) -> str:
    """Return label.

    Inputs:
        value: Candidate value to validate or transform.
        name: Parameter or field name used in validation messages.
    Outputs:
        A value of type `str` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    if not isinstance(value, str) or not value:
        raise ProcessQualityReleaseError(f"{name} must be a non-empty string.")
    return value


def _finite_number(
    value: Any,
    name: str,
    *,
    non_negative: bool,
) -> float:
    """Validate and return a finite numeric value.

    Inputs:
        value: Candidate value to validate or transform.
        name: Parameter or field name used in validation messages.
        non_negative: Input value used by this operation.
    Outputs:
        A value of type `float` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ProcessQualityReleaseError(f"{name} must be a finite numeric value.")
    numeric = float(value)
    if not math.isfinite(numeric) or (non_negative and numeric < 0):
        qualifier = " non-negative" if non_negative else ""
        raise ProcessQualityReleaseError(
            f"{name} must be a finite{qualifier} numeric value."
        )
    return numeric


def _count(value: Any, name: str) -> int:
    """Count.

    Inputs:
        value: Candidate value to validate or transform.
        name: Parameter or field name used in validation messages.
    Outputs:
        A value of type `int` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    numeric = _finite_number(value, name, non_negative=True)
    if not numeric.is_integer():
        raise ProcessQualityReleaseError(
            f"{name} must be a non-negative integer DP count."
        )
    return int(numeric)


def _probability(value: Any, name: str) -> float:
    """Validate and return a probability value.

    Inputs:
        value: Candidate value to validate or transform.
        name: Parameter or field name used in validation messages.
    Outputs:
        A value of type `float` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    probability = _finite_number(value, name, non_negative=True)
    if probability > 1.0:
        raise ProcessQualityReleaseError(f"{name} must be in [0, 1].")
    return probability


def _edge(record: Mapping[str, Any], name: str) -> Edge:
    """Return edge.

    Inputs:
        record: Single record processed by the operation.
        name: Parameter or field name used in validation messages.
    Outputs:
        A value of type `Edge` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    return (
        _label(_require(record, "source", name), f"{name}.source"),
        _label(_require(record, "target", name), f"{name}.target"),
    )


def _assert_exact(value: Any, expected: Any, name: str) -> None:
    """Check exact.

    Inputs:
        value: Candidate value to validate or transform.
        expected: Input value used by this operation.
        name: Parameter or field name used in validation messages.
    Outputs:
        None.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    if value != expected:
        raise ProcessQualityReleaseError(
            f"{name} differs from the authenticated checkpoint expectation: "
            f"expected {expected!r}, got {value!r}."
        )


def _validate_identity(
    metadata: Mapping[str, Any],
    *,
    expected_epsilon: float,
    expected_seed: int,
    frozen_t_max: float,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Validate identity.

    Inputs:
        metadata: Metadata associated with the artifact or experiment condition.
        expected_epsilon: Input value used by this operation.
        expected_seed: Input value used by this operation.
        frozen_t_max: Input value used by this operation.
    Outputs:
        A value of type `tuple[Mapping[str, Any], Mapping[str, Any]]` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    config = _mapping(_require(metadata, "config", "experiment_metadata"), "config")
    release = _mapping(
        _require(metadata, "release", "experiment_metadata"), "release"
    )

    epsilon = _finite_number(
        expected_epsilon, "expected_epsilon", non_negative=True
    )
    if epsilon == 0:
        raise ProcessQualityReleaseError("expected_epsilon must be strictly positive.")
    seed = _count(expected_seed, "expected_seed")
    if seed > 29:
        raise ProcessQualityReleaseError("expected_seed must be in the frozen range 0-29.")
    t_max = _finite_number(frozen_t_max, "frozen_t_max", non_negative=True)

    for container_name, container in (("config", config), ("release", release)):
        _assert_exact(
            _require(container, "method", container_name),
            "DP-full",
            f"{container_name}.method",
        )
        _assert_exact(
            _require(container, "budget_strategy", container_name),
            "uniform",
            f"{container_name}.budget_strategy",
        )
        if "dp_edge_mode" in container:
            _assert_exact(
                container["dp_edge_mode"], None, f"{container_name}.dp_edge_mode"
            )

    _assert_exact(
        _finite_number(
            _require(config, "epsilon", "config"),
            "config.epsilon",
            non_negative=True,
        ),
        epsilon,
        "config.epsilon",
    )
    for key in ("epsilon_total", "epsilon_spent"):
        _assert_exact(
            _finite_number(
                _require(release, key, "release"),
                f"release.{key}",
                non_negative=True,
            ),
            epsilon,
            f"release.{key}",
        )
    for container_name, container in (("config", config), ("release", release)):
        _assert_exact(
            _count(_require(container, "seed", container_name), f"{container_name}.seed"),
            seed,
            f"{container_name}.seed",
        )
    _assert_exact(
        _finite_number(
            _require(config, "t_max", "config"),
            "config.t_max",
            non_negative=True,
        ),
        t_max,
        "config.t_max",
    )
    return config, release


def _parse_dfg(release: Mapping[str, Any]) -> dict[Edge, int]:
    """Parse dfg.

    Inputs:
        release: Input value used by this operation.
    Outputs:
        A value of type `dict[Edge, int]` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    edge_counts: dict[Edge, int] = {}
    for index, value in enumerate(
        _records(_require(release, "dfg", "release"), "release.dfg")
    ):
        name = f"release.dfg[{index}]"
        record = _mapping(value, name)
        _exact_fields(record, _DFG_FIELDS, name)
        edge = _edge(record, name)
        if edge in edge_counts:
            raise ProcessQualityReleaseError(f"Duplicate DP DFG edge {edge!r}.")
        edge_counts[edge] = _count(record["count"], f"{name}.count")
        _probability(record["probability"], f"{name}.probability")
    return edge_counts


def _parse_temporal(
    release: Mapping[str, Any],
) -> tuple[dict[Edge, float], dict[Edge, float]]:
    """Parse temporal.

    Inputs:
        release: Input value used by this operation.
    Outputs:
        A value of type `tuple[dict[Edge, float], dict[Edge, float]]` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    time_sums: dict[Edge, float] = {}
    time_squared_sums: dict[Edge, float] = {}
    for index, value in enumerate(
        _records(_require(release, "temporal", "release"), "release.temporal")
    ):
        name = f"release.temporal[{index}]"
        record = _mapping(value, name)
        _exact_fields(record, _TEMPORAL_FIELDS, name)
        edge = _edge(record, name)
        if edge in time_sums:
            raise ProcessQualityReleaseError(f"Duplicate DP temporal edge {edge!r}.")
        time_sums[edge] = _finite_number(
            record["sum_seconds"], f"{name}.sum_seconds", non_negative=True
        )
        time_squared_sums[edge] = _finite_number(
            record["squared_sum_seconds2"],
            f"{name}.squared_sum_seconds2",
            non_negative=False,
        )
        _finite_number(
            record["mean_seconds"], f"{name}.mean_seconds", non_negative=True
        )
    return time_sums, time_squared_sums


def _parse_variants(
    release: Mapping[str, Any],
) -> tuple[dict[Variant, int], int]:
    """Parse variants.

    Inputs:
        release: Input value used by this operation.
    Outputs:
        A value of type `tuple[dict[Variant, int], int]` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    variant_counts: dict[Variant, int] = {}
    for index, value in enumerate(
        _records(_require(release, "variants", "release"), "release.variants")
    ):
        name = f"release.variants[{index}]"
        record = _mapping(value, name)
        _exact_fields(record, _VARIANT_FIELDS, name)
        activities = record["activities"]
        if not isinstance(activities, list) or not activities:
            raise ProcessQualityReleaseError(
                f"{name}.activities must be a non-empty JSON list."
            )
        variant = tuple(
            _label(activity, f"{name}.activities[{position}]")
            for position, activity in enumerate(activities)
        )
        if "OTHER" in variant:
            raise ProcessQualityReleaseError(
                "OTHER must remain separate from concrete DP Variants."
            )
        if variant in variant_counts:
            raise ProcessQualityReleaseError(f"Duplicate DP Variant {variant!r}.")
        variant_counts[variant] = _count(record["count"], f"{name}.count")
        _probability(record["probability"], f"{name}.probability")

    other = _mapping(_require(release, "variant_other", "release"), "release.variant_other")
    _exact_fields(other, _OTHER_FIELDS, "release.variant_other")
    _assert_exact(
        _label(other["label"], "release.variant_other.label"),
        "OTHER",
        "release.variant_other.label",
    )
    other_count = _count(other["count"], "release.variant_other.count")
    _probability(other["probability"], "release.variant_other.probability")
    return variant_counts, other_count


def _parse_sample_counts(value: Any) -> dict[Edge, int]:
    """Parse sample counts.

    Inputs:
        value: Candidate value to validate or transform.
    Outputs:
        A value of type `dict[Edge, int]` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    result: dict[Edge, int] = {}
    for index, item in enumerate(_records(value, "temporal_generation.sample_counts")):
        name = f"temporal_generation.sample_counts[{index}]"
        record = _mapping(item, name)
        _exact_fields(
            record,
            frozenset({"source", "target", "sample_count"}),
            name,
        )
        edge = _edge(record, name)
        if edge in result:
            raise ProcessQualityReleaseError(
                f"Duplicate checkpoint temporal sample-count edge {edge!r}."
            )
        result[edge] = _count(record["sample_count"], f"{name}.sample_count")
    return result


def _temporal_edge_science(value: Any, name: str) -> tuple[Edge, dict[str, Any]]:
    """Return temporal edge science.

    Inputs:
        value: Candidate value to validate or transform.
        name: Parameter or field name used in validation messages.
    Outputs:
        A value of type `tuple[Edge, dict[str, Any]]` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    record = _mapping(value, name)
    edge = _edge(record, name)
    science: dict[str, Any] = {}
    for key in _TEMPORAL_EDGE_SCIENCE_FIELDS:
        science[key] = _require(record, key, name)
    return edge, science


def _verify_checkpoint_temporal_generation(
    metadata: Mapping[str, Any],
    generated: TemporalDistributionResult,
) -> None:
    """Verify checkpoint temporal generation.

    Inputs:
        metadata: Metadata associated with the artifact or experiment condition.
        generated: Input value used by this operation.
    Outputs:
        None.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    checkpoint = _mapping(
        _require(metadata, "temporal_generation", "experiment_metadata"),
        "temporal_generation",
    )
    _assert_exact(
        _count(_require(checkpoint, "seed", "temporal_generation"), "temporal_generation.seed"),
        generated.seed,
        "temporal_generation.seed",
    )
    _assert_exact(
        _finite_number(
            _require(checkpoint, "t_max", "temporal_generation"),
            "temporal_generation.t_max",
            non_negative=True,
        ),
        generated.t_max,
        "temporal_generation.t_max",
    )
    _assert_exact(
        _finite_number(
            _require(checkpoint, "delta", "temporal_generation"),
            "temporal_generation.delta",
            non_negative=True,
        ),
        generated.delta,
        "temporal_generation.delta",
    )
    checkpoint_counts = _parse_sample_counts(
        _require(checkpoint, "sample_counts", "temporal_generation")
    )
    _assert_exact(
        checkpoint_counts,
        generated.sample_counts,
        "temporal_generation.sample_counts",
    )

    checkpoint_metadata = _mapping(
        _require(checkpoint, "metadata", "temporal_generation"),
        "temporal_generation.metadata",
    )
    for key in _TEMPORAL_SCIENCE_FIELDS:
        _assert_exact(
            _require(checkpoint_metadata, key, "temporal_generation.metadata"),
            generated.metadata[key],
            f"temporal_generation.metadata.{key}",
        )

    checkpoint_edges: dict[Edge, dict[str, Any]] = {}
    for index, value in enumerate(
        _records(
            _require(checkpoint_metadata, "edges", "temporal_generation.metadata"),
            "temporal_generation.metadata.edges",
        )
    ):
        edge, science = _temporal_edge_science(
            value, f"temporal_generation.metadata.edges[{index}]"
        )
        if edge in checkpoint_edges:
            raise ProcessQualityReleaseError(
                f"Duplicate checkpoint temporal-science edge {edge!r}."
            )
        checkpoint_edges[edge] = science

    generated_edges: dict[Edge, dict[str, Any]] = {}
    for index, value in enumerate(generated.metadata["edges"]):
        edge, science = _temporal_edge_science(
            value, f"generated_temporal.metadata.edges[{index}]"
        )
        generated_edges[edge] = science
    _assert_exact(
        checkpoint_edges,
        generated_edges,
        "temporal_generation.metadata.edges",
    )


def generate_process_quality_release(
    experiment_metadata: Mapping[str, Any],
    *,
    expected_epsilon: float,
    expected_seed: int,
    frozen_t_max: float,
) -> ProcessQualityReleaseResult:
    """Rebuild a DP-consistent synthetic log from authenticated metadata.

    Input
    -----
    experiment_metadata:
        An already authenticated ``experiment_metadata.json`` dictionary.  The
        bridge accepts no raw trace, event log, evaluation log, or raw samples.
    expected_epsilon, expected_seed:
        Caller-frozen formal-condition identity.  Both the config and DP
        release must match these values and the fixed DP-full/uniform method.
    frozen_t_max:
        Caller-frozen transition-time clipping bound.  The checkpoint value is
        checked against it, but the caller value drives temporal reconstruction.

    Output
    ------
    ProcessQualityReleaseResult
        Strict DP mappings, regenerated ``T'_e``, and the synthetic event log.

    Paper experiment step
    ---------------------
    Phase 6 external process-quality bridge: authenticated DP statistics are
    deserialized into ``C'_E``, ``S'_T``, ``Q'_T`` and concrete ``C'_V``;
    existing ``TemporalDistributionGenerator`` and ``SyntheticLogGenerator``
    perform DP-only post-processing.  ``OTHER`` is recorded but excluded.
    """

    metadata = _mapping(experiment_metadata, "experiment_metadata")
    _, release = _validate_identity(
        metadata,
        expected_epsilon=expected_epsilon,
        expected_seed=expected_seed,
        frozen_t_max=frozen_t_max,
    )
    edge_counts = _parse_dfg(release)
    time_sums, time_squared_sums = _parse_temporal(release)
    if set(edge_counts) != set(time_sums) or set(edge_counts) != set(
        time_squared_sums
    ):
        raise ProcessQualityReleaseError(
            "release.dfg and release.temporal must use exactly the same "
            "frozen DP edge domain."
        )
    variant_counts, other_count = _parse_variants(release)

    temporal_distributions = TemporalDistributionGenerator(
        t_max=frozen_t_max,
        seed=expected_seed,
    ).generate(
        edge_counts=edge_counts,
        time_sums=time_sums,
        time_squared_sums=time_squared_sums,
    )
    _verify_checkpoint_temporal_generation(metadata, temporal_distributions)

    # MissingTemporalEdgeError intentionally propagates unchanged from the
    # established generator; this bridge supplies no fallback edge or time.
    synthetic_log = SyntheticLogGenerator(seed=expected_seed).generate(
        variant_counts,
        temporal_distributions,
        other_count=other_count,
    )
    bridge_metadata: dict[str, Any] = {
        "name": "BUL_process_quality_DP_release",
        "module": "generate_process_quality_release",
        "status": "success",
        "dp_post_processing_only": True,
        "raw_data_access": False,
        "authenticated_metadata_required": True,
        "method": "DP-full",
        "budget_strategy": "uniform",
        "epsilon": float(expected_epsilon),
        "seed": int(expected_seed),
        "t_max": float(frozen_t_max),
        "edge_domain_count": len(edge_counts),
        "concrete_variant_count": len(variant_counts),
        "concrete_case_count": synthetic_log.case_count,
        "excluded_other_count": other_count,
        "other_policy": "excluded_from_synthetic_generation",
        "temporal_checkpoint_science_verified": True,
    }
    return ProcessQualityReleaseResult(
        edge_counts=edge_counts,
        time_sums=time_sums,
        time_squared_sums=time_squared_sums,
        variant_counts=variant_counts,
        other_count=other_count,
        temporal_distributions=temporal_distributions,
        synthetic_log=synthetic_log,
        metadata=bridge_metadata,
    )


__all__ = [
    "ProcessQualityReleaseError",
    "ProcessQualityReleaseResult",
    "generate_process_quality_release",
]
