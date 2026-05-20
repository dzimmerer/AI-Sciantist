"""Tests for sciantist.metrics module."""

from __future__ import annotations

import math

import pytest

from sciantist.metrics import (
    _as_finite_float,
    _metric_is_better,
    _metric_matches_or_better,
    calc_unified_metric,
    extract_weighted_metric_inputs,
)


class TestCalcUnifiedMetric:
    """Test calc_unified_metric function."""

    def test_single_metric(self) -> None:
        metrics = {"val/x": 0.8}
        weights = {"val/x": 1.0}
        result = calc_unified_metric(metrics, weights)
        assert result == 0.8

    def test_multiple_metrics_equal_weights(self) -> None:
        metrics = {"val/x": 0.8, "val/y": 0.6}
        weights = {"val/x": 0.5, "val/y": 0.5}
        result = calc_unified_metric(metrics, weights)
        assert result == 0.7

    def test_multiple_metrics_unequal_weights(self) -> None:
        metrics = {"val/x": 0.8, "val/y": 0.4}
        weights = {"val/x": 0.75, "val/y": 0.25}
        result = calc_unified_metric(metrics, weights)
        assert result == pytest.approx(0.7)

    def test_missing_metric_treated_as_zero(self) -> None:
        metrics = {"val/x": 0.8}
        weights = {"val/x": 0.5, "val/y": 0.5}
        result = calc_unified_metric(metrics, weights)
        assert result == pytest.approx(0.8)

    def test_zero_weight_sum_returns_nan(self) -> None:
        metrics = {"val/x": 0.8}
        weights = {}
        result = calc_unified_metric(metrics, weights)
        assert math.isnan(result)

    def test_non_float_values_ignored(self) -> None:
        metrics = {"val/x": "not a number", "val/y": 0.8}
        weights = {"val/x": 0.5, "val/y": 0.5}
        result = calc_unified_metric(metrics, weights)
        assert result == pytest.approx(0.8)

    def test_integer_weights(self) -> None:
        metrics = {"val/x": 10, "val/y": 20}
        weights = {"val/x": 1, "val/y": 1}
        result = calc_unified_metric(metrics, weights)
        assert result == 15.0


class TestAsFiniteFloat:
    """Test _as_finite_float function."""

    def test_integer_returns_float(self) -> None:
        assert _as_finite_float(42) == 42.0

    def test_float_returns_float(self) -> None:
        assert _as_finite_float(3.14) == 3.14

    def test_none_returns_none(self) -> None:
        assert _as_finite_float(None) is None

    def test_string_returns_none(self) -> None:
        assert _as_finite_float("42") is None

    def test_bool_returns_none(self) -> None:
        assert _as_finite_float(True) is None
        assert _as_finite_float(False) is None

    def test_inf_returns_none(self) -> None:
        assert _as_finite_float(float("inf")) is None
        assert _as_finite_float(float("-inf")) is None

    def test_nan_returns_none(self) -> None:
        assert _as_finite_float(float("nan")) is None


class TestMetricIsBetter:
    """Test _metric_is_better function."""

    def test_none_new_returns_false(self) -> None:
        assert _metric_is_better(None, 0.5, True) is False

    def test_none_baseline_returns_true(self) -> None:
        assert _metric_is_better(0.5, None, True) is True

    def test_higher_is_better_new_higher(self) -> None:
        assert _metric_is_better(0.9, 0.8, True) is True

    def test_higher_is_better_new_lower(self) -> None:
        assert _metric_is_better(0.7, 0.8, True) is False

    def test_lower_is_better_new_lower(self) -> None:
        assert _metric_is_better(0.7, 0.8, False) is True

    def test_lower_is_better_new_higher(self) -> None:
        assert _metric_is_better(0.9, 0.8, False) is False

    def test_equal_is_not_better(self) -> None:
        assert _metric_is_better(0.8, 0.8, True) is False


class TestMetricMatchesOrBetter:
    """Test _metric_matches_or_better function."""

    def test_none_reference_returns_true(self) -> None:
        assert _metric_matches_or_better(0.5, None, True) is True

    def test_none_metric_returns_false(self) -> None:
        assert _metric_matches_or_better(None, 0.5, True) is False

    def test_higher_is_better_higher_match(self) -> None:
        assert _metric_matches_or_better(0.9, 0.8, True) is True

    def test_higher_is_better_exact_match(self) -> None:
        assert _metric_matches_or_better(0.8, 0.8, True) is True

    def test_higher_is_better_lower_fails(self) -> None:
        assert _metric_matches_or_better(0.7, 0.8, True) is False

    def test_lower_is_better_lower_match(self) -> None:
        assert _metric_matches_or_better(0.7, 0.8, False) is True

    def test_lower_is_better_exact_match(self) -> None:
        assert _metric_matches_or_better(0.8, 0.8, False) is True

    def test_lower_is_better_higher_fails(self) -> None:
        assert _metric_matches_or_better(0.9, 0.8, False) is False


class TestExtractWeightedMetricInputs:
    """Test extract_weighted_metric_inputs function."""

    def test_extracts_from_summary(self) -> None:
        wandb_payload = {
            "summary": {"val/x": 0.8, "val/y": 0.6},
            "last_step_metrics": {},
            "metric_histories": {},
        }
        metric_weights = {"val/x": 0.5, "val/y": 0.5}
        metrics, histories = extract_weighted_metric_inputs(wandb_payload, metric_weights)
        assert metrics["val/x"] == 0.8
        assert metrics["val/y"] == 0.6

    def test_falls_back_to_last_row(self) -> None:
        wandb_payload = {
            "summary": {},
            "last_step_metrics": {"val/x": 0.75},
            "metric_histories": {},
        }
        metric_weights = {"val/x": 1.0}
        metrics, histories = extract_weighted_metric_inputs(wandb_payload, metric_weights)
        assert metrics["val/x"] == 0.75

    def test_falls_back_to_history_last_value(self) -> None:
        wandb_payload = {
            "summary": {},
            "last_step_metrics": {},
            "metric_histories": {"val/x": [0.3, 0.5, 0.7]},
        }
        metric_weights = {"val/x": 1.0}
        metrics, histories = extract_weighted_metric_inputs(wandb_payload, metric_weights)
        assert metrics["val/x"] == 0.7

    def test_extracts_histories(self) -> None:
        wandb_payload = {
            "summary": {"val/x": 0.8},
            "last_step_metrics": {},
            "metric_histories": {"val/x": [0.3, 0.5, 0.7, 0.8]},
        }
        metric_weights = {"val/x": 1.0}
        metrics, histories = extract_weighted_metric_inputs(wandb_payload, metric_weights)
        assert histories["val/x"] == [0.3, 0.5, 0.7, 0.8]

    def test_non_dict_payloads_return_empty(self) -> None:
        wandb_payload = "not a dict"
        metric_weights = {"val/x": 1.0}
        try:
            metrics, histories = extract_weighted_metric_inputs(wandb_payload, metric_weights)
        except (AttributeError, TypeError):
            pass
        else:
            assert metrics == {}
            assert histories == {}