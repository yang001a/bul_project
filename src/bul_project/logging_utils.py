"""Shared logging configuration for release entry points.

功能：
    为所有论文运行脚本配置一致的控制台与文件日志。
输入：
    日志名称、可选日志目录和日志级别。
输出：
    已配置的 ``logging.Logger``。
对应论文章节：
    Reproducibility and implementation details.
数学定义：
    不适用；本模块不参与任何数值计算。
"""

from __future__ import annotations

import logging
from pathlib import Path

from .config import LOG_ROOT


LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | "
    "dataset=%(dataset)s | epsilon=%(epsilon)s | seed=%(seed)s | "
    "method=%(method)s | %(message)s"
)


class _ContextDefaults(logging.Filter):
    """Supply stable empty experiment fields for non-condition log messages.

    功能：
        为未携带实验上下文的日志记录补充四个标准字段。
    输入：
        ``logging.LogRecord``。
    输出：
        总是返回 ``True``，允许记录继续输出。
    对应论文章节：
        Reproducibility and implementation details.
    数学定义：
        不适用。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Populate missing dataset, epsilon, seed, and method fields.

        功能：
            保证统一日志格式不会因缺少上下文字段而失败。
        输入：
            一条日志记录。
        输出：
            ``True``。
        对应论文章节：
            Reproducibility and implementation details.
        数学定义：
            不适用。
        """

        for name in ("dataset", "epsilon", "seed", "method"):
            if not hasattr(record, name):
                setattr(record, name, "-")
        return True


def configure_logging(
    log_name: str,
    *,
    log_dir: str | Path = LOG_ROOT,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure the release-wide logging format and return one logger.

    功能：
        同时启用 UTF-8 文件日志和控制台日志。
    输入：
        日志名称、日志目录和日志级别。
    输出：
        可直接写入实验上下文的 logger。
    对应论文章节：
        Reproducibility and implementation details.
    数学定义：
        不适用；不改变实验计算。
    """

    if not isinstance(log_name, str) or not log_name.strip():
        raise ValueError("log_name must be a non-empty string.")
    destination = Path(log_dir)
    destination.mkdir(parents=True, exist_ok=True)
    context_filter = _ContextDefaults()
    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = logging.FileHandler(
        destination / f"{log_name.strip()}.log",
        encoding="utf-8",
    )
    stream_handler = logging.StreamHandler()
    for handler in (file_handler, stream_handler):
        handler.setFormatter(formatter)
        handler.addFilter(context_filter)

    logging.basicConfig(
        level=level,
        handlers=[file_handler, stream_handler],
        force=True,
    )
    return logging.getLogger(log_name.strip())


def experiment_context(
    *,
    dataset: object = "-",
    epsilon: object = "-",
    seed: object = "-",
    method: object = "-",
) -> dict[str, object]:
    """Build the standard structured logging context.

    功能：
        生成 dataset、epsilon、seed 和 method 四字段上下文。
    输入：
        四个实验标识值。
    输出：
        可传给 ``logging`` 的 ``extra`` 字典。
    对应论文章节：
        Reproducibility and implementation details.
    数学定义：
        不适用。
    """

    return {
        "dataset": dataset,
        "epsilon": epsilon,
        "seed": seed,
        "method": method,
    }
