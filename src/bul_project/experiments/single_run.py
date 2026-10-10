"""Phase 7A orchestration for one reproducible BUL experiment condition."""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from bul_project.behavior import Edge, GRawStatistics, Variant, build_g_raw
from bul_project.bul import (
    calculate_BUL,
    calculate_information_loss,
    calculate_structure_loss,
    calculate_temporal_loss,
)
from bul_project.config import (
    BUDGET_STRATEGIES,
    DP_EDGE_MODES,
    EPS_LIST,
    STRUCTURE_COMPONENT_WEIGHTS,
)
from bul_project.metrics import calculate_cosine_loss, calculate_js
from bul_project.logging_utils import experiment_context
from bul_project.preprocessing.models import Trace
from bul_project.privacy import (
    DPReleaseStatistics,
    clip_traces,
    dp_edge_only_release,
    dp_full_release,
)
from bul_project.privacy._validation import (
    validate_epsilon,
    validate_experiment_seed,
    validate_l_max,
    validate_t_max,
)
from bul_project.privacy.temporal_generator import (
    TemporalDistributionGenerator,
    TemporalDistributionResult,
)

from bul_project.validation.process_model import (
    ConformanceVariant,
    ProcessModelEvaluator,
    ProcessQualityResult,
)
from bul_project.validation.synthetic_log import (
    SyntheticLogGenerator,
    SyntheticLogResult,
)


ExperimentMethod = Literal["DP-full", "DP-edge-only"]
_OTHER_VARIANT = "OTHER"
_STAGE_NAMES = ("clipping", "dp_noise", "total")
LOGGER = logging.getLogger(__name__)


def _validate_positive_integer(value: int, name: str) -> int:
    """Validate positive integer.

    Inputs:
        value: Candidate value to validate or transform.
        name: Parameter or field name used in validation messages.
    Outputs:
        A value of type `int` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a strictly positive integer.")
    return value


def _validate_noise_threshold(value: float) -> float:
    """Validate noise threshold.

    Inputs:
        value: Candidate value to validate or transform.
    Outputs:
        A value of type `float` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            "process_model_noise_threshold must be finite and in [0,1]."
        )
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0 or numeric > 1:
        raise ValueError(
            "process_model_noise_threshold must be finite and in [0,1]."
        )
    return numeric


@dataclass(frozen=True)
class SingleExperimentConfig:
    """Explicit configuration for exactly one Phase 7A experiment condition."""

    dataset: str
    method: ExperimentMethod
    epsilon: float
    seed: int
    l_max: int
    t_max: float
    top_k_variants: int
    min_edge_support: int
    budget_strategy: str | None
    dp_edge_mode: str | None
    evaluate_process_quality: bool
    conformance_variant: ConformanceVariant | None
    process_model_noise_threshold: float | None

    def __post_init__(self) -> None:
        """Validate and normalize the dataclass state after initialization.

        Inputs:
            None.
        Outputs:
            None.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        if not isinstance(self.dataset, str) or not self.dataset.strip():
            raise ValueError("dataset must be a non-empty string.")
        if self.method not in ("DP-full", "DP-edge-only"):
            raise ValueError("method must be either 'DP-full' or 'DP-edge-only'.")
        if not isinstance(self.evaluate_process_quality, bool):
            raise ValueError("evaluate_process_quality must be bool.")

        object.__setattr__(self, "dataset", self.dataset.strip())
        epsilon = validate_epsilon(self.epsilon, "epsilon")
        if epsilon not in EPS_LIST:
            raise ValueError(
                f"epsilon must be one of the configured values {EPS_LIST}."
            )
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(
            self,
            "seed",
            validate_experiment_seed(self.seed),
        )
        object.__setattr__(self, "l_max", validate_l_max(self.l_max))
        object.__setattr__(self, "t_max", validate_t_max(self.t_max))
        object.__setattr__(
            self,
            "top_k_variants",
            _validate_positive_integer(
                self.top_k_variants,
                "top_k_variants",
            ),
        )
        object.__setattr__(
            self,
            "min_edge_support",
            _validate_positive_integer(
                self.min_edge_support,
                "min_edge_support",
            ),
        )

        if self.method == "DP-full":
            if self.budget_strategy not in BUDGET_STRATEGIES:
                raise ValueError(
                    "DP-full requires one configured budget_strategy."
                )
            if self.dp_edge_mode is not None:
                raise ValueError("DP-full must not set dp_edge_mode.")
        else:
            if self.budget_strategy is not None:
                raise ValueError(
                    "DP-edge-only must not set a DP-full budget_strategy."
                )
            if self.dp_edge_mode not in DP_EDGE_MODES:
                raise ValueError(
                    "DP-edge-only requires component or mechanism mode."
                )
            if self.evaluate_process_quality:
                raise ValueError(
                    "DP-edge-only cannot run log-level process quality "
                    "without DP Variant and temporal releases."
                )

        if self.evaluate_process_quality:
            if self.conformance_variant not in (
                "token_based",
                "alignments",
            ):
                raise ValueError(
                    "Process-quality evaluation requires an explicit "
                    "conformance_variant."
                )
            if self.process_model_noise_threshold is None:
                raise ValueError(
                    "Process-quality evaluation requires an explicit "
                    "process_model_noise_threshold."
                )
            object.__setattr__(
                self,
                "process_model_noise_threshold",
                _validate_noise_threshold(
                    self.process_model_noise_threshold
                ),
            )
        elif (
            self.conformance_variant is not None
            or self.process_model_noise_threshold is not None
        ):
            raise ValueError(
                "Disabled process-quality evaluation must not carry unused "
                "conformance parameters."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible parameters for the single experiment."""

        return {
            "dataset": self.dataset,
            "method": self.method,
            "epsilon": self.epsilon,
            "seed": self.seed,
            "l_max": self.l_max,
            "t_max": self.t_max,
            "top_k_variants": self.top_k_variants,
            "min_edge_support": self.min_edge_support,
            "budget_strategy": self.budget_strategy,
            "dp_edge_mode": self.dp_edge_mode,
            "evaluate_process_quality": self.evaluate_process_quality,
            "conformance_variant": self.conformance_variant,
            "process_model_noise_threshold": (
                self.process_model_noise_threshold
            ),
        }


@dataclass(frozen=True)
class LossDimensions:
    """One independently calculated clipping, DP-noise, or total loss."""

    l_structure: float
    l_temporal: float | None
    l_information: float | None
    bul: float | None

    def to_dict(self) -> dict[str, float | None]:
        """Return the four required result fields without filling missing data."""

        return {
            "L_structure": self.l_structure,
            "L_temporal": self.l_temporal,
            "L_information": self.l_information,
            "BUL": self.bul,
        }


@dataclass(frozen=True)
class SingleExperimentResult:
    """Complete in-memory result for one Phase 7A condition."""

    config: SingleExperimentConfig
    stage_losses: dict[str, LossDimensions]
    release: DPReleaseStatistics
    temporal_distributions: TemporalDistributionResult | None
    synthetic_log: SyntheticLogResult | None
    process_quality: ProcessQualityResult | None
    runtime_seconds: float
    analysis_metadata: dict[str, Any]

    @property
    def total_losses(self) -> LossDimensions:
        """Return the G_raw versus released-result comparison."""

        return self.stage_losses["total"]

    def to_record(self) -> dict[str, Any]:
        """Return one flat row with the required experiment-result columns."""

        quality = self.process_quality
        losses = self.total_losses
        return {
            "dataset": self.config.dataset,
            "epsilon": self.config.epsilon,
            "method": self.config.method,
            "seed": self.config.seed,
            "L_structure": losses.l_structure,
            "L_temporal": losses.l_temporal,
            "L_information": losses.l_information,
            "BUL": losses.bul,
            "fitness": None if quality is None else quality.fitness,
            "precision": None if quality is None else quality.precision,
            "generalization": (
                None if quality is None else quality.generalization
            ),
            "runtime": self.runtime_seconds,
            "l_max": self.config.l_max,
            "t_max": self.config.t_max,
            "top_k_variants": self.config.top_k_variants,
            "min_edge_support": self.config.min_edge_support,
            "budget_strategy": self.config.budget_strategy,
            "dp_edge_mode": self.config.dp_edge_mode,
            "process_quality_evaluated": (
                self.config.evaluate_process_quality
            ),
            "conformance_variant": self.config.conformance_variant,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return complete JSON-compatible results and reproducibility metadata."""

        temporal = None
        if self.temporal_distributions is not None:
            temporal = {
                "seed": self.temporal_distributions.seed,
                "t_max": self.temporal_distributions.t_max,
                "delta": self.temporal_distributions.delta,
                "sample_counts": [
                    {
                        "source": edge[0],
                        "target": edge[1],
                        "sample_count": count,
                    }
                    for edge, count in sorted(
                        self.temporal_distributions.sample_counts.items()
                    )
                ],
                "metadata": self.temporal_distributions.metadata,
            }

        synthetic = None
        if self.synthetic_log is not None:
            synthetic = {
                "seed": self.synthetic_log.seed,
                "case_count": self.synthetic_log.case_count,
                "event_count": len(self.synthetic_log.events),
                "metadata": self.synthetic_log.metadata,
            }

        return {
            "name": "BUL_single_experiment",
            "phase": "7A",
            "config": self.config.to_dict(),
            "result_record": self.to_record(),
            "stage_losses": {
                stage: self.stage_losses[stage].to_dict()
                for stage in _STAGE_NAMES
            },
            "release": self.release.to_dict(),
            "temporal_generation": temporal,
            "synthetic_log_generation": synthetic,
            "process_quality": (
                None
                if self.process_quality is None
                else self.process_quality.to_dict()
            ),
            "analysis_metadata": self.analysis_metadata,
        }

    def metadata_json(self) -> str:
        """Serialize the complete single-run result as deterministic JSON."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


def _project_counts(
    counts: Mapping[Any, int | float],
    domain: Sequence[Any],
) -> dict[Any, float]:
    """Project counts.

    Inputs:
        counts: Input value used by this operation.
        domain: Input value used by this operation.
    Outputs:
        A value of type `dict[Any, float]` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    return {key: float(counts.get(key, 0.0)) for key in domain}


def _variant_vector(
    statistics: GRawStatistics,
    candidate_variants: Sequence[Variant],
) -> dict[Variant | str, float]:
    """Return variant vector.

    Inputs:
        statistics: Input value used by this operation.
        candidate_variants: Input value used by this operation.
    Outputs:
        A value of type `dict[Variant | str, float]` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    vector: dict[Variant | str, float] = {
        variant: float(statistics.variants.full_counts.get(variant, 0))
        for variant in candidate_variants
    }
    other_count = statistics.variants.total_cases - sum(vector.values())
    if other_count < 0:
        raise RuntimeError("Projected Variant OTHER count became negative.")
    vector[_OTHER_VARIANT] = float(other_count)
    return vector


def _released_variant_vector(
    release: DPReleaseStatistics,
    candidate_variants: Sequence[Variant],
) -> dict[Variant | str, float]:
    """Return released variant vector.

    Inputs:
        release: Input value used by this operation.
        candidate_variants: Input value used by this operation.
    Outputs:
        A value of type `dict[Variant | str, float]` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    if release.variant_probabilities is None:
        raise RuntimeError("DP-full release is missing Variant probabilities.")
    if release.variant_other_probability is None:
        raise RuntimeError("DP-full release is missing the Variant OTHER bucket.")
    vector: dict[Variant | str, float] = {
        variant: float(release.variant_probabilities.get(variant, 0.0))
        for variant in candidate_variants
    }
    vector[_OTHER_VARIANT] = float(release.variant_other_probability)
    return vector


def _effective_temporal_edges(
    g_raw: GRawStatistics,
    g_clip: GRawStatistics,
    edge_domain: Sequence[Edge],
) -> tuple[Edge, ...]:
    """Return effective temporal edges.

    Inputs:
        g_raw: Input value used by this operation.
        g_clip: Input value used by this operation.
        edge_domain: Input value used by this operation.
    Outputs:
        A value of type `tuple[Edge, ...]` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    return tuple(
        edge
        for edge in edge_domain
        if g_raw.temporal.samples.get(edge)
        and g_clip.temporal.samples.get(edge)
    )


def _fixed_temporal_inputs(
    g_raw: GRawStatistics,
    effective_edges: Sequence[Edge],
    normalization_scales: Mapping[Edge, float] | None,
) -> tuple[dict[Edge, float], dict[Edge, float]]:
    """Return fixed temporal inputs.

    Inputs:
        g_raw: Input value used by this operation.
        effective_edges: Input value used by this operation.
        normalization_scales: Input value used by this operation.
    Outputs:
        A value of type `tuple[dict[Edge, float], dict[Edge, float]]` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    if not effective_edges:
        raise ValueError(
            "The frozen edge domain has no edge with valid raw and clipped "
            "transition-time samples."
        )
    if normalization_scales is None:
        raise ValueError(
            "DP-full requires caller-supplied, epsilon-independent "
            "normalization_scales."
        )
    missing = set(effective_edges) - set(normalization_scales)
    if missing:
        raise ValueError(
            "normalization_scales is missing frozen temporal edges: "
            f"{missing!r}"
        )

    scales: dict[Edge, float] = {}
    for edge in effective_edges:
        value = normalization_scales[edge]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(
                f"normalization_scales[{edge!r}] must be finite and non-negative."
            )
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError(
                f"normalization_scales[{edge!r}] must be finite and non-negative."
            )
        scales[edge] = numeric

    masses = {
        edge: float(g_raw.dfg.counts.get(edge, 0))
        for edge in effective_edges
    }
    total_mass = sum(masses.values())
    if total_mass <= 0:
        raise RuntimeError("Raw effective-edge probability mass must be positive.")
    weights = {
        edge: masses[edge] / total_mass for edge in effective_edges
    }
    return weights, scales


def _full_loss_dimensions(
    *,
    reference_activities: Mapping[Any, float],
    comparison_activities: Mapping[Any, float],
    reference_edges: Mapping[Any, float],
    comparison_edges: Mapping[Any, float],
    reference_variants: Mapping[Any, float],
    comparison_variants: Mapping[Any, float],
    reference_times: Mapping[Edge, Sequence[float]],
    comparison_times: Mapping[Edge, Sequence[float]],
    temporal_edge_weights: Mapping[Edge, float],
    normalization_scales: Mapping[Edge, float],
) -> LossDimensions:
    """Return full loss dimensions.

    Inputs:
        reference_activities: Input value used by this operation.
        comparison_activities: Input value used by this operation.
        reference_edges: Input value used by this operation.
        comparison_edges: Input value used by this operation.
        reference_variants: Input value used by this operation.
        comparison_variants: Input value used by this operation.
        reference_times: Input value used by this operation.
        comparison_times: Input value used by this operation.
        temporal_edge_weights: Input value used by this operation.
        normalization_scales: Input value used by this operation.
    Outputs:
        A value of type `LossDimensions` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    l_structure = calculate_structure_loss(
        reference_edges,
        comparison_edges,
        reference_variants,
        comparison_variants,
    )
    l_temporal = calculate_temporal_loss(
        reference_times,
        comparison_times,
        edge_weights=temporal_edge_weights,
        normalization_scales=normalization_scales,
    )
    l_information = calculate_information_loss(
        reference_activities,
        comparison_activities,
        reference_edges,
        comparison_edges,
    )
    return LossDimensions(
        l_structure=l_structure,
        l_temporal=l_temporal,
        l_information=l_information,
        bul=calculate_BUL(
            l_structure,
            l_temporal,
            l_information,
        ),
    )


def _edge_only_structure_loss(
    reference_edges: Mapping[Any, float],
    comparison_edges: Mapping[Any, float],
) -> float:
    """Return edge only structure loss.

    Inputs:
        reference_edges: Input value used by this operation.
        comparison_edges: Input value used by this operation.
    Outputs:
        A value of type `float` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    available_weight = (
        STRUCTURE_COMPONENT_WEIGHTS["edge_js"]
        + STRUCTURE_COMPONENT_WEIGHTS["edge_cosine_loss"]
    )
    if available_weight <= 0:
        raise RuntimeError(
            "Configured edge-only structure weights must have positive mass."
        )
    edge_js_weight = (
        STRUCTURE_COMPONENT_WEIGHTS["edge_js"] / available_weight
    )
    edge_cosine_weight = (
        STRUCTURE_COMPONENT_WEIGHTS["edge_cosine_loss"]
        / available_weight
    )
    return (
        edge_js_weight * calculate_js(reference_edges, comparison_edges)
        + edge_cosine_weight
        * calculate_cosine_loss(reference_edges, comparison_edges)
    )


def _edge_only_dimensions(
    reference_edges: Mapping[Any, float],
    comparison_edges: Mapping[Any, float],
) -> LossDimensions:
    """Return edge only dimensions.

    Inputs:
        reference_edges: Input value used by this operation.
        comparison_edges: Input value used by this operation.
    Outputs:
        A value of type `LossDimensions` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    return LossDimensions(
        l_structure=_edge_only_structure_loss(
            reference_edges,
            comparison_edges,
        ),
        l_temporal=None,
        l_information=None,
        bul=None,
    )


class SingleExperimentRunner:
    """Connect the approved Phase 1 through Phase 6 APIs for one condition."""

    def __init__(self, config: SingleExperimentConfig) -> None:
        """Initialize the instance and validate its construction inputs.

        Inputs:
            config: Configuration controlling the operation.
        Outputs:
            None.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        if not isinstance(config, SingleExperimentConfig):
            raise TypeError("config must be a SingleExperimentConfig.")
        self.config = config

    def run(
        self,
        traces: Sequence[Trace],
        *,
        normalization_scales: Mapping[Edge, float] | None = None,
        evaluation_events: pd.DataFrame | None = None,
        evaluation_log_id: str | None = None,
    ) -> SingleExperimentResult:
        """Run one DP release, loss evaluation, and optional process validation.

        Input
        -----
        traces:
            Non-empty Phase 1 traces for one explicitly named dataset.
        normalization_scales:
            Epsilon-independent per-edge nEMD scales fixed before this run.
            Required for DP-full and unused by DP-edge-only.
        evaluation_events, evaluation_log_id:
            One explicit unified evaluation log and identifier. Both are
            required exactly when process-quality evaluation is enabled.

        Output
        ------
        SingleExperimentResult
            Total, clipping-only and DP-noise-only losses; DP metadata;
            optional synthetic-log process quality; and total runtime.

        Paper experiment step
        ---------------------
        Phase 7A applies Steps 3-14 once: construct G_raw and G_clip, release
        one G_DP, independently compare clipping, DP noise and total loss, and
        optionally perform Scheme A external validation from the DP synthetic
        log. It does not perform batch repetition or statistical tests.
        """

        if self.config.method == "DP-full":
            if normalization_scales is None:
                raise ValueError(
                    "DP-full requires fixed normalization_scales."
                )
        elif normalization_scales is not None:
            raise ValueError(
                "DP-edge-only must not receive unused temporal scales."
            )

        if self.config.evaluate_process_quality:
            if evaluation_events is None:
                raise ValueError(
                    "Process-quality evaluation requires evaluation_events."
                )
            if (
                not isinstance(evaluation_log_id, str)
                or not evaluation_log_id.strip()
            ):
                raise ValueError(
                    "Process-quality evaluation requires evaluation_log_id."
                )
        elif evaluation_events is not None or evaluation_log_id is not None:
            raise ValueError(
                "Disabled process-quality evaluation must not receive an "
                "unused evaluation log."
            )

        log_context = experiment_context(
            dataset=self.config.dataset,
            epsilon=self.config.epsilon,
            seed=self.config.seed,
            method=self.config.method,
        )
        LOGGER.info("Starting reproducible experiment condition.", extra=log_context)

        started = time.perf_counter()
        materialized_traces = tuple(traces)
        g_raw = build_g_raw(materialized_traces)
        clipped_traces = clip_traces(
            materialized_traces,
            self.config.l_max,
            self.config.t_max,
        )
        g_clip = build_g_raw(
            clipped_traces,
            top_k_variants=self.config.top_k_variants,
            min_edge_support=self.config.min_edge_support,
        )

        edge_domain = tuple(sorted(g_clip.dfg.filtered_counts))
        if not edge_domain:
            raise ValueError(
                "min_edge_support removed every edge from the frozen domain."
            )
        activity_domain = tuple(sorted(g_clip.activity.counts))
        candidate_variants = tuple(g_clip.variants.top_counts)

        raw_edges = _project_counts(g_raw.dfg.counts, edge_domain)
        clipped_edges = _project_counts(
            g_clip.dfg.filtered_counts,
            edge_domain,
        )

        temporal_result: TemporalDistributionResult | None = None
        synthetic_result: SyntheticLogResult | None = None
        process_result: ProcessQualityResult | None = None

        if self.config.method == "DP-full":
            if self.config.budget_strategy is None:
                raise RuntimeError("Validated DP-full strategy is unavailable.")
            release = dp_full_release(
                g_clip,
                l_max=self.config.l_max,
                t_max=self.config.t_max,
                epsilon_total=self.config.epsilon,
                strategy=self.config.budget_strategy,
                seed=self.config.seed,
            )
            if (
                release.activity_probabilities is None
                or release.variant_counts is None
                or release.time_sums is None
                or release.time_squared_sums is None
            ):
                raise RuntimeError(
                    "DP-full release omitted a required statistic family."
                )

            effective_edges = _effective_temporal_edges(
                g_raw,
                g_clip,
                edge_domain,
            )
            temporal_edge_weights, fixed_scales = _fixed_temporal_inputs(
                g_raw,
                effective_edges,
                normalization_scales,
            )
            raw_times = {
                edge: g_raw.temporal.samples[edge]
                for edge in effective_edges
            }
            clipped_times = {
                edge: g_clip.temporal.samples[edge]
                for edge in effective_edges
            }

            temporal_result = TemporalDistributionGenerator(
                t_max=self.config.t_max,
                seed=self.config.seed,
            ).generate(
                edge_counts=release.edge_counts,
                time_sums=release.time_sums,
                time_squared_sums=release.time_squared_sums,
            )
            dp_times = {
                edge: temporal_result.samples[edge]
                for edge in effective_edges
            }

            raw_activities = _project_counts(
                g_raw.activity.counts,
                activity_domain,
            )
            clipped_activities = _project_counts(
                g_clip.activity.counts,
                activity_domain,
            )
            dp_activities = _project_counts(
                release.activity_probabilities,
                activity_domain,
            )
            raw_variants = _variant_vector(
                g_raw,
                candidate_variants,
            )
            clipped_variants = _variant_vector(
                g_clip,
                candidate_variants,
            )
            dp_variants = _released_variant_vector(
                release,
                candidate_variants,
            )

            stage_losses = {
                "clipping": _full_loss_dimensions(
                    reference_activities=raw_activities,
                    comparison_activities=clipped_activities,
                    reference_edges=raw_edges,
                    comparison_edges=clipped_edges,
                    reference_variants=raw_variants,
                    comparison_variants=clipped_variants,
                    reference_times=raw_times,
                    comparison_times=clipped_times,
                    temporal_edge_weights=temporal_edge_weights,
                    normalization_scales=fixed_scales,
                ),
                "dp_noise": _full_loss_dimensions(
                    reference_activities=clipped_activities,
                    comparison_activities=dp_activities,
                    reference_edges=clipped_edges,
                    comparison_edges=release.edge_probabilities,
                    reference_variants=clipped_variants,
                    comparison_variants=dp_variants,
                    reference_times=clipped_times,
                    comparison_times=dp_times,
                    temporal_edge_weights=temporal_edge_weights,
                    normalization_scales=fixed_scales,
                ),
                "total": _full_loss_dimensions(
                    reference_activities=raw_activities,
                    comparison_activities=dp_activities,
                    reference_edges=raw_edges,
                    comparison_edges=release.edge_probabilities,
                    reference_variants=raw_variants,
                    comparison_variants=dp_variants,
                    reference_times=raw_times,
                    comparison_times=dp_times,
                    temporal_edge_weights=temporal_edge_weights,
                    normalization_scales=fixed_scales,
                ),
            }

            if self.config.evaluate_process_quality:
                synthetic_result = SyntheticLogGenerator(
                    seed=self.config.seed
                ).generate(
                    release.variant_counts,
                    temporal_result,
                    other_count=release.variant_other_count or 0,
                )
                if (
                    self.config.conformance_variant is None
                    or self.config.process_model_noise_threshold is None
                    or evaluation_events is None
                    or evaluation_log_id is None
                ):
                    raise RuntimeError(
                        "Validated process-quality inputs are unavailable."
                    )
                process_result = ProcessModelEvaluator(
                    conformance_variant=self.config.conformance_variant,
                    noise_threshold=(
                        self.config.process_model_noise_threshold
                    ),
                ).evaluate(
                    synthetic_result,
                    evaluation_events,
                    evaluation_log_id=evaluation_log_id,
                )

            temporal_metadata = [
                {
                    "source": edge[0],
                    "target": edge[1],
                    "raw_edge_weight": temporal_edge_weights[edge],
                    "normalization_scale": fixed_scales[edge],
                }
                for edge in effective_edges
            ]
            analysis_metadata = {
                "comparison_stages": {
                    "clipping": "G_raw versus G_clip",
                    "dp_noise": "G_clip versus G_DP",
                    "total": "G_raw versus G_DP",
                    "additivity_claimed": False,
                },
                "statistical_spaces_frozen_before_noise": True,
                "activity_domain": list(activity_domain),
                "edge_domain": [
                    {"source": edge[0], "target": edge[1]}
                    for edge in edge_domain
                ],
                "variant_domain": [
                    list(variant) for variant in candidate_variants
                ]
                + [_OTHER_VARIANT],
                "temporal_domain": temporal_metadata,
                "normalization_scale_policy": (
                    "caller_supplied_fixed_before_epsilon_run"
                ),
                "temporal_weight_policy": (
                    "raw_edge_probability_renormalized_over_E_m"
                ),
                "process_quality_policy": (
                    "DP_synthetic_log_scheme_A"
                    if self.config.evaluate_process_quality
                    else "not_requested_for_this_condition"
                ),
            }
        else:
            if self.config.dp_edge_mode is None:
                raise RuntimeError(
                    "Validated DP-edge-only mode is unavailable."
                )
            release = dp_edge_only_release(
                g_clip,
                l_max=self.config.l_max,
                t_max=self.config.t_max,
                epsilon_total=self.config.epsilon,
                seed=self.config.seed,
                mode=self.config.dp_edge_mode,
            )
            stage_losses = {
                "clipping": _edge_only_dimensions(
                    raw_edges,
                    clipped_edges,
                ),
                "dp_noise": _edge_only_dimensions(
                    clipped_edges,
                    release.edge_probabilities,
                ),
                "total": _edge_only_dimensions(
                    raw_edges,
                    release.edge_probabilities,
                ),
            }
            available_weight = (
                STRUCTURE_COMPONENT_WEIGHTS["edge_js"]
                + STRUCTURE_COMPONENT_WEIGHTS["edge_cosine_loss"]
            )
            analysis_metadata = {
                "comparison_stages": {
                    "clipping": "G_raw edge vector versus G_clip edge vector",
                    "dp_noise": "G_clip edge vector versus G_DP edge vector",
                    "total": "G_raw edge vector versus G_DP edge vector",
                    "additivity_claimed": False,
                },
                "statistical_spaces_frozen_before_noise": True,
                "edge_domain": [
                    {"source": edge[0], "target": edge[1]}
                    for edge in edge_domain
                ],
                "edge_only_structure_components": {
                    "edge_js": (
                        STRUCTURE_COMPONENT_WEIGHTS["edge_js"]
                        / available_weight
                    ),
                    "edge_cosine_loss": (
                        STRUCTURE_COMPONENT_WEIGHTS[
                            "edge_cosine_loss"
                        ]
                        / available_weight
                    ),
                },
                "missing_dimension_policy": (
                    "L_temporal, L_information, BUL, fitness, precision, "
                    "and generalization are NA"
                ),
                "raw_variant_or_time_substitution": False,
                "process_quality_policy": (
                    "not_computable_from_edge_count_only"
                ),
            }

        runtime_seconds = time.perf_counter() - started
        LOGGER.info(
            "Completed reproducible experiment condition in %.6f seconds.",
            runtime_seconds,
            extra=log_context,
        )
        return SingleExperimentResult(
            config=self.config,
            stage_losses=stage_losses,
            release=release,
            temporal_distributions=temporal_result,
            synthetic_log=synthetic_result,
            process_quality=process_result,
            runtime_seconds=runtime_seconds,
            analysis_metadata=analysis_metadata,
        )


def run_single_experiment(
    traces: Sequence[Trace],
    config: SingleExperimentConfig,
    *,
    normalization_scales: Mapping[Edge, float] | None = None,
    evaluation_events: pd.DataFrame | None = None,
    evaluation_log_id: str | None = None,
) -> SingleExperimentResult:
    """Run one Phase 7A condition through the explicit runner.

    Input
    -----
    traces, config:
        Phase 1 traces and one validated experiment configuration.
    normalization_scales:
        Pre-fixed nEMD scales required by DP-full.
    evaluation_events, evaluation_log_id:
        Explicit unified process-evaluation log inputs when enabled.

    Output
    ------
    SingleExperimentResult
        The same typed result returned by SingleExperimentRunner.run.

    Paper experiment step
    ---------------------
    Phase 7A single-condition orchestration only. Batch epsilon/seed loops and
    statistical significance analysis remain outside this function.
    """

    return SingleExperimentRunner(config).run(
        traces,
        normalization_scales=normalization_scales,
        evaluation_events=evaluation_events,
        evaluation_log_id=evaluation_log_id,
    )


def write_single_experiment_artifacts(
    result: SingleExperimentResult,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write one flat CSV row and its complete metadata JSON.

    Input
    -----
    result:
        One completed SingleExperimentResult.
    output_dir:
        Destination directory for the two Phase 7A artifacts.

    Output
    ------
    dict[str, str]
        Artifact names for experiment_result.csv and
        experiment_metadata.json.

    Paper experiment step
    ---------------------
    Phase 7A records all required result fields, seed, parameters, stage
    comparisons and generation metadata. Missing edge-only dimensions are
    serialized as NA; no batch append or statistical analysis occurs here.
    """

    if not isinstance(result, SingleExperimentResult):
        raise TypeError("result must be a SingleExperimentResult.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    csv_path = destination / "experiment_result.csv"
    temporary_csv = csv_path.with_name(f".{csv_path.name}.tmp")
    pd.DataFrame([result.to_record()]).to_csv(
        temporary_csv,
        index=False,
        encoding="utf-8",
        na_rep="NA",
    )
    temporary_csv.replace(csv_path)

    metadata_path = destination / "experiment_metadata.json"
    temporary_metadata = metadata_path.with_name(
        f".{metadata_path.name}.tmp"
    )
    temporary_metadata.write_text(
        result.metadata_json() + "\n",
        encoding="utf-8",
    )
    temporary_metadata.replace(metadata_path)
    return {
        "experiment_result": csv_path.name,
        "experiment_metadata": metadata_path.name,
    }
