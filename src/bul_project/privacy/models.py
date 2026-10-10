"""Typed data contracts for Phase 3 clipping and DP releases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bul_project.behavior.models import Edge, Variant


@dataclass(frozen=True)
class SensitivityBounds:
    """Conservative vector L1 sensitivities under case-level adjacency."""

    activity: float
    edge: float
    variant: float
    time: float
    time_squared: float
    l_max: int
    t_max: float

    @property
    def time_sum(self) -> float:
        """Return the first-moment sensitivity Delta_ST."""

        return self.time

    @property
    def time_squared_sum(self) -> float:
        """Return the second-moment sensitivity Delta_QT."""

        return self.time_squared

    def to_dict(self) -> dict[str, float | int]:
        """Return JSON-compatible sensitivity metadata."""

        return {
            "activity": self.activity,
            "edge": self.edge,
            "variant": self.variant,
            "time": self.time,
            "time_sum": self.time_sum,
            "time_squared_sum": self.time_squared_sum,
            "l_max": self.l_max,
            "t_max": self.t_max,
        }


@dataclass(frozen=True)
class PrivacyBudget:
    """One sequentially composed allocation across four statistic families."""

    epsilon_total: float
    strategy: str
    activity: float
    edge: float
    variant: float
    time: float

    @property
    def composed_epsilon(self) -> float:
        """Return epsilon_A + epsilon_E + epsilon_V + epsilon_T."""

        return self.activity + self.edge + self.variant + self.time

    def to_dict(self) -> dict[str, float | str]:
        """Return JSON-compatible allocation metadata."""

        return {
            "strategy": self.strategy,
            "epsilon_total": self.epsilon_total,
            "epsilon_activity": self.activity,
            "epsilon_edge": self.edge,
            "epsilon_variant": self.variant,
            "epsilon_time": self.time,
            "composed_epsilon": self.composed_epsilon,
            "composition": "sequential",
        }


@dataclass(frozen=True)
class DPReleaseStatistics:
    """A reproducible ``G_DP`` result for DP-full or DP-edge-only."""

    method: str
    epsilon_total: float
    epsilon_spent: float
    component_epsilons: dict[str, float]
    temporal_epsilons: dict[str, float] | None
    budget_strategy: str | None
    dp_edge_mode: str | None
    seed: int
    sensitivities: SensitivityBounds
    top_k_variants: int | None
    min_edge_support: int | None
    activity_counts: dict[str, int] | None
    activity_probabilities: dict[str, float] | None
    edge_counts: dict[Edge, int]
    edge_probabilities: dict[Edge, float]
    variant_counts: dict[Variant, int] | None
    variant_probabilities: dict[Variant, float] | None
    variant_other_count: int | None
    variant_other_probability: float | None
    time_sums: dict[Edge, float] | None
    time_squared_sums: dict[Edge, float] | None
    time_means: dict[Edge, float] | None

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation of ``G_DP``."""

        activity = None
        if self.activity_counts is not None and self.activity_probabilities is not None:
            activity = [
                {
                    "activity": key,
                    "count": self.activity_counts[key],
                    "probability": self.activity_probabilities[key],
                }
                for key in sorted(self.activity_counts)
            ]

        edges = [
            {
                "source": edge[0],
                "target": edge[1],
                "count": self.edge_counts[edge],
                "probability": self.edge_probabilities[edge],
            }
            for edge in sorted(self.edge_counts)
        ]

        variants = None
        if self.variant_counts is not None and self.variant_probabilities is not None:
            variants = [
                {
                    "activities": list(variant),
                    "count": self.variant_counts[variant],
                    "probability": self.variant_probabilities[variant],
                }
                for variant in sorted(self.variant_counts)
            ]

        temporal = None
        if (
            self.time_sums is not None
            and self.time_squared_sums is not None
            and self.time_means is not None
        ):
            temporal = [
                {
                    "source": edge[0],
                    "target": edge[1],
                    "sum_seconds": self.time_sums[edge],
                    "squared_sum_seconds2": self.time_squared_sums[edge],
                    "mean_seconds": self.time_means[edge],
                }
                for edge in sorted(self.time_sums)
            ]

        other = None
        if self.variant_other_count is not None:
            other = {
                "label": "OTHER",
                "count": self.variant_other_count,
                "probability": self.variant_other_probability,
            }

        return {
            "name": "G_DP",
            "method": self.method,
            "epsilon_total": self.epsilon_total,
            "epsilon_spent": self.epsilon_spent,
            "composition": "sequential",
            "component_epsilons": dict(self.component_epsilons),
            "temporal_epsilons": (
                None
                if self.temporal_epsilons is None
                else {
                    **dict(self.temporal_epsilons),
                    "composed_epsilon_time": sum(
                        self.temporal_epsilons.values()
                    ),
                    "composition": "sequential",
                }
            ),
            "budget_strategy": self.budget_strategy,
            "dp_edge_mode": self.dp_edge_mode,
            "seed": self.seed,
            "sensitivities": self.sensitivities.to_dict(),
            "release_domain": {
                "source": "G_clip; frozen before Laplace noise",
                "top_k_variants": self.top_k_variants,
                "min_edge_support": self.min_edge_support,
            },
            "activity": activity,
            "dfg": edges,
            "variants": variants,
            "variant_other": other,
            "temporal": temporal,
        }
