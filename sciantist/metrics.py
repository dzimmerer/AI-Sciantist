"""Metric comparison and W&B extraction helpers."""

from __future__ import annotations

import math
from typing import Any


def calc_unified_metric(metrics: dict[str, Any], weights: dict[str, float]) -> float:
    """Compute weighted-sum unified metric from a W&B metrics dictionary."""
    total = 0.0
    weight_sum = 0.0
    for metric_name, weight in weights.items():
        value = metrics.get(metric_name)
        if isinstance(value, (int, float)):
            total += float(value) * float(weight)
            weight_sum += float(weight)
    if weight_sum == 0.0:
        return float("nan")
    return total / weight_sum


def _as_finite_float(value: Any) -> float | None:
    """Return value as finite float when possible."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    return result


def extract_weighted_metric_inputs(
    wandb_payload: dict[str, Any],
    metric_weights: dict[str, float],
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Build scalar metrics and full histories for configured weighted metrics."""
    summary = wandb_payload.get("summary")
    last_row = wandb_payload.get("last_step_metrics")
    history_payload = wandb_payload.get("metric_histories")

    summary_dict = summary if isinstance(summary, dict) else {}
    last_row_dict = last_row if isinstance(last_row, dict) else {}
    history_dict = history_payload if isinstance(history_payload, dict) else {}

    metrics_for_unified: dict[str, float] = {}
    metric_histories: dict[str, list[float]] = {}
    for metric_name in metric_weights.keys():
        history_values: list[float] = []
        raw_history = history_dict.get(metric_name)
        if isinstance(raw_history, list):
            for raw_value in raw_history:
                numeric_value = _as_finite_float(raw_value)
                if numeric_value is not None:
                    history_values.append(numeric_value)
        metric_histories[metric_name] = history_values

        scalar_value = _as_finite_float(summary_dict.get(metric_name))
        if scalar_value is None:
            scalar_value = _as_finite_float(last_row_dict.get(metric_name))
        if scalar_value is None and history_values:
            scalar_value = history_values[-1]
        if scalar_value is not None:
            metrics_for_unified[metric_name] = scalar_value

    return metrics_for_unified, metric_histories


def _metric_is_better(new_metric: float, baseline_metric: float | None, higher_is_better: bool) -> bool:
    """Return True if new metric should be accepted against baseline."""
    finite_new_metric = _as_finite_float(new_metric)
    if finite_new_metric is None:
        return False

    finite_baseline_metric = _as_finite_float(baseline_metric)
    if finite_baseline_metric is None:
        return True
    if higher_is_better:
        return finite_new_metric > finite_baseline_metric
    return finite_new_metric < finite_baseline_metric


def _metric_matches_or_better(metric: float, reference_metric: float | None, higher_is_better: bool) -> bool:
    """Return True if metric is at least as good as the reference metric."""
    finite_metric = _as_finite_float(metric)
    if finite_metric is None:
        return False

    finite_reference_metric = _as_finite_float(reference_metric)
    if finite_reference_metric is None:
        return True
    if higher_is_better:
        return finite_metric >= finite_reference_metric
    return finite_metric <= finite_reference_metric
