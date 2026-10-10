"""Deterministic case-level sample for the Phase 6 appendix evaluation log.

The sample is used only to score the three representative PM4Py models.  It is
never supplied to the DP release, temporal reconstruction, or synthetic-log
generation path.
"""

from __future__ import annotations

import copy
import hashlib
from numbers import Integral
from typing import Any

import pandas as pd

from bul_project.experiments.parameter_freeze import build_evaluation_log_metadata


EVALUATION_SAMPLE_POLICY_VERSION = "BUL_PM4Py_evaluation_case_sample_v1"
EVALUATION_SAMPLE_SEED = 27
EVALUATION_SAMPLE_CASES = 3000
EVALUATION_SAMPLE_LOG_ID = "BPIC2017_fixed_hash_case_sample_seed27_n3000"
EVALUATION_SAMPLE_SHA256 = (
    "62003efe9a4237efa84bc454db12888f48c0ed37ea7474df48173398ee1805e0"
)
EVALUATION_SAMPLE_CASE_SET_SHA256 = (
    "212d0c186ef7ee59f8b02423442321bef36a7e7db8e7e89e087cf4f25455fa0f"
)
EVALUATION_SAMPLE_EVENT_COUNT = 114955

_CORE_COLUMNS = ("case_id", "activity", "timestamp")


class EvaluationSampleIntegrityError(RuntimeError):
    """Raised when the deterministic evaluation sample contract is violated."""


def _case_score(case_id: str, *, seed: int, policy_version: str) -> str:
    """Return case score.

    Inputs:
        case_id: Input value used by this operation.
        seed: Random seed controlling reproducible execution.
        policy_version: Input value used by this operation.
    Outputs:
        A value of type `str` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    payload = f"{policy_version}|seed={seed}|case_id={case_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deterministic_case_sample(
    events: pd.DataFrame,
    *,
    sample_size_cases: int,
    seed: int,
    policy_version: str = EVALUATION_SAMPLE_POLICY_VERSION,
) -> pd.DataFrame:
    """Select a reproducible case sample by SHA-256 rank without replacement.

    Input
    -----
    events:
        Canonical event table with ``case_id``, ``activity``, and ``timestamp``.
        The table is an external evaluation log, not a DP-generation input.
    sample_size_cases:
        Exact number of whole cases to retain.
    seed:
        Frozen experiment seed included in every case-ranking payload.
    policy_version:
        Stable identifier included in the SHA-256 ranking payload.

    Output
    ------
    pandas.DataFrame
        A copy containing every event of the selected cases, in input row order.

    Paper experiment step
    ---------------------
    Phase 6 appendix external-validity check: create one fixed case-level
    evaluation subset before calculating Fitness, Precision, or Generalization.
    The identical subset is reused for all three representative epsilon values.
    """

    if not isinstance(events, pd.DataFrame):
        raise TypeError("events must be a pandas DataFrame.")
    missing = set(_CORE_COLUMNS) - set(events.columns)
    if missing:
        raise ValueError(f"events is missing canonical columns: {missing!r}")
    if isinstance(sample_size_cases, bool) or not isinstance(
        sample_size_cases, Integral
    ):
        raise TypeError("sample_size_cases must be a positive integer.")
    sample_size = int(sample_size_cases)
    if sample_size <= 0:
        raise ValueError("sample_size_cases must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer.")
    if not isinstance(policy_version, str) or not policy_version.strip():
        raise ValueError("policy_version must be a non-empty string.")

    case_ids = events["case_id"].drop_duplicates().tolist()
    if any(not isinstance(case_id, str) or not case_id for case_id in case_ids):
        raise ValueError("events contains an invalid case_id.")
    if sample_size > len(case_ids):
        raise ValueError(
            "sample_size_cases cannot exceed the evaluation-log case count."
        )

    ranked = sorted(
        case_ids,
        key=lambda case_id: (
            _case_score(
                case_id,
                seed=int(seed),
                policy_version=policy_version.strip(),
            ),
            case_id,
        ),
    )
    selected = set(ranked[:sample_size])
    sampled = (
        events.loc[events["case_id"].isin(selected), list(_CORE_COLUMNS)]
        .copy()
        .reset_index(drop=True)
    )
    if int(sampled["case_id"].nunique()) != sample_size:
        raise EvaluationSampleIntegrityError(
            "The deterministic sampler did not retain the requested case count."
        )
    return sampled


def expected_evaluation_sample_metadata() -> dict[str, Any]:
    """Return the immutable BPIC2017 sample fingerprint and selection policy."""

    return copy.deepcopy(
        {
            "evaluation_log_id": EVALUATION_SAMPLE_LOG_ID,
            "evaluation_log_sha256": EVALUATION_SAMPLE_SHA256,
            "event_count": EVALUATION_SAMPLE_EVENT_COUNT,
            "case_count": EVALUATION_SAMPLE_CASES,
            "core_columns": list(_CORE_COLUMNS),
            "timestamp_policy": "complete canonical timestamps required",
            "condition_policy": (
                "same event rows and fingerprint for every epsilon, method, "
                "and seed in this dataset"
            ),
        }
    )


def expected_evaluation_sampling_policy() -> dict[str, Any]:
    """Return the immutable case-selection rule used by the appendix check."""

    return copy.deepcopy(
        {
            "policy_version": EVALUATION_SAMPLE_POLICY_VERSION,
            "sampling_unit": "case_id",
            "method": "SHA256_rank_without_replacement",
            "seed": EVALUATION_SAMPLE_SEED,
            "sample_size_cases": EVALUATION_SAMPLE_CASES,
            "population_case_count": 31509,
            "population_event_count": 1202267,
            "score_payload_template": (
                f"{EVALUATION_SAMPLE_POLICY_VERSION}|seed="
                f"{EVALUATION_SAMPLE_SEED}|case_id={{case_id}}"
            ),
            "selected_case_id_sha256": EVALUATION_SAMPLE_CASE_SET_SHA256,
            "same_sample_for_all_conditions": True,
            "role": "external_scoring_only_not_DP_generation",
        }
    )


def build_fixed_bpic2017_evaluation_sample(events: pd.DataFrame) -> pd.DataFrame:
    """Build and authenticate the frozen 3000-case BPIC2017 evaluation sample.

    Input
    -----
    events:
        Complete canonical BPIC2017 evaluation event table whose source file has
        already been authenticated by the caller.

    Output
    ------
    pandas.DataFrame
        The exact frozen case-level sample accepted by the Phase 6 PM4Py runner.

    Paper experiment step
    ---------------------
    Phase 6 appendix external validity: deterministic resource-bounded sampling
    of the common evaluation log before model scoring.  No sampled row enters
    DP reconstruction or synthetic-log generation.
    """

    sampled = deterministic_case_sample(
        events,
        sample_size_cases=EVALUATION_SAMPLE_CASES,
        seed=EVALUATION_SAMPLE_SEED,
    )
    actual_metadata = build_evaluation_log_metadata(
        sampled, EVALUATION_SAMPLE_LOG_ID
    ).to_dict()
    if actual_metadata != expected_evaluation_sample_metadata():
        raise EvaluationSampleIntegrityError(
            "The BPIC2017 evaluation sample differs from its frozen fingerprint."
        )
    selected_case_ids = sorted(sampled["case_id"].drop_duplicates().tolist())
    selected_hash = hashlib.sha256(
        "\n".join(selected_case_ids).encode("utf-8")
    ).hexdigest()
    if selected_hash != EVALUATION_SAMPLE_CASE_SET_SHA256:
        raise EvaluationSampleIntegrityError(
            "The BPIC2017 evaluation case set differs from its frozen fingerprint."
        )
    return sampled


__all__ = [
    "EVALUATION_SAMPLE_CASES",
    "EVALUATION_SAMPLE_LOG_ID",
    "EVALUATION_SAMPLE_POLICY_VERSION",
    "EVALUATION_SAMPLE_SEED",
    "EvaluationSampleIntegrityError",
    "build_fixed_bpic2017_evaluation_sample",
    "deterministic_case_sample",
    "expected_evaluation_sample_metadata",
    "expected_evaluation_sampling_policy",
]
