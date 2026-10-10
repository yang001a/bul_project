"""Central configuration for the SCI Artifact release.

功能：
    集中管理数据路径、冻结实验参数、隐私预算、随机种子和数值稳定常量。
输入：
    可选环境变量 ``BUL_DATASET_ROOT``。
输出：
    供各功能模块只读使用的配置常量和数据集说明。
对应论文章节：
    Experimental setup and reproducibility.
数学定义：
    本模块只保存冻结参数，不实现或修改任何数学公式。
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = SOURCE_ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DATASET_ROOT = Path(
    os.environ.get("BUL_DATASET_ROOT", str(PROJECT_ROOT / "data" / "raw"))
).expanduser()
PROCESSED_DATA_ROOT = PROJECT_ROOT / "data" / "processed"
LOG_ROOT = PROJECT_ROOT / "logs"
CONFIG_ROOT = PROJECT_ROOT / "configs"
RESULTS_ROOT = PROJECT_ROOT / "results"
# The public artifact ships without data-derived configuration artifacts.
# ``freeze-parameters`` generates this local bundle from the user's XES files.
GENERATED_CONFIG_DIR = CONFIG_ROOT / "generated"
FROZEN_CONFIG_PATH = GENERATED_CONFIG_DIR / "experiment_config_final.json"
FROZEN_PARAMETER_DIR = GENERATED_CONFIG_DIR
MAIN_EXPERIMENT_RESULT_DIR = RESULTS_ROOT / "main_experiment"
DERIVED_ANALYSIS_RESULT_DIR = RESULTS_ROOT / "derived_analysis"
SENSITIVITY_ANALYSIS_RESULT_DIR = RESULTS_ROOT / "sensitivity_analysis"
EXTERNAL_VALIDATION_RESULT_DIR = RESULTS_ROOT / "external_validation"
STATISTICAL_ANALYSIS_RESULT_DIR = RESULTS_ROOT / "statistical_analysis"

REPRODUCED_MAIN_EXPERIMENT_DIR = MAIN_EXPERIMENT_RESULT_DIR
REPRODUCED_PARAMETER_DIR = GENERATED_CONFIG_DIR
REPRODUCED_DERIVED_ANALYSIS_DIR = DERIVED_ANALYSIS_RESULT_DIR
REPRODUCED_SENSITIVITY_ANALYSIS_DIR = SENSITIVITY_ANALYSIS_RESULT_DIR
REPRODUCED_EXTERNAL_VALIDATION_DIR = EXTERNAL_VALIDATION_RESULT_DIR
REPRODUCED_STATISTICAL_ANALYSIS_DIR = STATISTICAL_ANALYSIS_RESULT_DIR

CORE_COLUMNS = ("case_id", "activity", "timestamp")
SUPPORTED_INPUT_SUFFIXES = (".xes", ".xes.gz")

# Phase 2 exposes the experiment-plan candidate values without selecting a
# main-experiment default prematurely. The runner in a later approved phase
# must choose one value and record it in its experiment configuration.
TOP_K_VARIANT_OPTIONS = (50, 100, 200)
EDGE_TAU_OPTIONS = (5, 10, 20)
FROZEN_TOP_K_VARIANTS = 100
FROZEN_EDGE_TAU = 10

# Phase 3 case-level differential-privacy experiment parameters.  Ratios are
# ordered as activity, edge, variant, and time; every row therefore represents
# sequential composition over the same case population and sums to one.
EPS_LIST = (0.5, 1.0, 2.0, 4.0, 8.0)
EPSILON_VALUES = EPS_LIST
LMAX_QUANTILE = 0.95
TMAX_QUANTILE = 0.95
EPS_MIN = 0.01
EPSILON_MIN = EPS_MIN
REPEAT_SEEDS = 30
EXPERIMENT_SEEDS = tuple(range(REPEAT_SEEDS))
PRIVACY_COMPONENTS = ("activity", "edge", "variant", "time")
PRIVACY_BUDGET_RATIOS: dict[str, dict[str, float]] = {
    "uniform": {
        "activity": 0.25,
        "edge": 0.25,
        "variant": 0.25,
        "time": 0.25,
    },
    "activity_focus": {
        "activity": 0.40,
        "edge": 0.20,
        "variant": 0.20,
        "time": 0.20,
    },
    "edge_focus": {
        "activity": 0.20,
        "edge": 0.40,
        "variant": 0.20,
        "time": 0.20,
    },
    "variant_focus": {
        "activity": 0.20,
        "edge": 0.20,
        "variant": 0.40,
        "time": 0.20,
    },
    "time_focus": {
        "activity": 0.20,
        "edge": 0.20,
        "variant": 0.20,
        "time": 0.40,
    },
}
BUDGET_STRATEGIES = tuple(PRIVACY_BUDGET_RATIOS)
TEMPORAL_BUDGET_COMPONENTS = ("time_sum", "time_squared_sum")
TEMPORAL_BUDGET_RATIOS: dict[str, float] = {
    "time_sum": 0.5,
    "time_squared_sum": 0.5,
}
DP_EDGE_MODES = ("component", "mechanism")
DP_EDGE_MODE = "component"
DEFAULT_DP_EDGE_MODE = DP_EDGE_MODE
DP_EDGE_COMPONENT_SHARE = 0.25
DP_DENOMINATOR_DELTA = 1e-8

# Phase 4 metric stability constants.  These protect the published formulas;
# they do not introduce alternative distance definitions.
DELTA_PROB = 1e-10
DELTA_DENOM = 1e-8
NEMD_ZERO_SCALE = 1.0

# Phase 5 transparent aggregation defaults.  Base metric formulas remain in
# metrics/; these weights only combine already loss-directed components.
STRUCTURE_COMPONENT_WEIGHTS = {
    "edge_js": 1.0 / 3.0,
    "edge_cosine_loss": 1.0 / 3.0,
    "variant_js": 1.0 / 3.0,
}
INFORMATION_COMPONENT_WEIGHTS = {
    "activity_entropy_loss": 1.0 / 3.0,
    "edge_entropy_loss": 1.0 / 3.0,
    "activity_js": 1.0 / 3.0,
}
BUL_DIMENSION_WEIGHTS = {
    "structure": 1.0 / 3.0,
    "temporal": 1.0 / 3.0,
    "information": 1.0 / 3.0,
}
WEIGHT_SUM_TOLERANCE = 1e-12

# Frozen compact robustness and representative external-validation scopes.
SENSITIVITY_DATASET = "BPIC2017"
SENSITIVITY_METHOD = "DP-full"
SENSITIVITY_BUDGET_STRATEGY = "uniform"
SENSITIVITY_EPSILON_VALUES = (0.5, 2.0, 8.0)
SENSITIVITY_TOP_K_OPTIONS = TOP_K_VARIANT_OPTIONS
SENSITIVITY_EDGE_TAU_OPTIONS = EDGE_TAU_OPTIONS
EXTERNAL_VALIDATION_DATASET = "BPIC2017"
EXTERNAL_VALIDATION_METHOD = "DP-full"
EXTERNAL_VALIDATION_BUDGET_STRATEGY = "uniform"
EXTERNAL_VALIDATION_EPSILON_VALUES = SENSITIVITY_EPSILON_VALUES

# Diagnostic, PM4Py-smoke, and statistical-analysis command defaults.
PARAMETER_SELECTION_SCOPE = "global"
VARIANT_COVERAGE_DIAGNOSTIC_THRESHOLD = 0.95
EDGE_MASS_DIAGNOSTIC_THRESHOLD = 0.95
TIME_SAMPLE_DIAGNOSTIC_THRESHOLD = 0.95
PROCESS_QUALITY_DATASETS = ("BPIC2017",)
PROCESS_CONFORMANCE_VARIANT = "token_based"
INDUCTIVE_MINER_NOISE_THRESHOLD = 0.0
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0


@dataclass(frozen=True)
class DatasetSpec:
    """Mapping from one XES dataset to the canonical Phase 1 schema."""

    name: str
    path: Path
    case_column: str = "case:concept:name"
    activity_column: str = "concept:name"
    timestamp_column: str = "time:timestamp"
    role: str = "validation"

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable dictionary representation.

        Inputs:
            None.
        Outputs:
            A value of type `dict[str, str]` produced by the operation.
        Paper role:
            Centralized experiment configuration and parameter freezing.
        """
        data = asdict(self)
        data["path"] = str(self.path)
        return data


@dataclass(frozen=True)
class PreprocessingConfig:
    """Rules whose values must be recorded with every preprocessing run."""

    keep_extensions: bool = False
    drop_exact_duplicates: bool = False
    timestamp_utc: bool = True
    missing_timestamp_position: str = "last"
    stable_sort_kind: str = "mergesort"
    transition_time_unit: str = "seconds"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation.

        Inputs:
            None.
        Outputs:
            A value of type `dict[str, object]` produced by the operation.
        Paper role:
            Centralized experiment configuration and parameter freezing.
        """
        return asdict(self)


DATASETS: dict[str, DatasetSpec] = {
    "BPIC2017": DatasetSpec(
        name="BPIC2017",
        path=DATASET_ROOT / "BPI_Challenge_2017.xes",
        role="main",
    ),
    "BPIC2012": DatasetSpec(
        name="BPIC2012",
        path=DATASET_ROOT / "BPI_Challenge_2012.xes",
        role="same-domain robustness",
    ),
    "SEPSIS": DatasetSpec(
        name="SEPSIS",
        path=DATASET_ROOT / "Sepsis_Cases.xes",
        role="cross-domain validation",
    ),
}

DEFAULT_PREPROCESSING = PreprocessingConfig()


def get_dataset_spec(name: str) -> DatasetSpec:
    """Return a configured dataset by its case-insensitive project name."""

    key = name.strip().upper()
    aliases = {
        "BPIC17": "BPIC2017",
        "BPIC2017": "BPIC2017",
        "BPIC12": "BPIC2012",
        "BPIC2012": "BPIC2012",
        "SEPSIS": "SEPSIS",
        "SEPSIS CASES": "SEPSIS",
    }
    try:
        return DATASETS[aliases[key]]
    except KeyError as exc:
        valid = ", ".join(DATASETS)
        raise KeyError(f"Unknown dataset {name!r}. Expected one of: {valid}.") from exc
