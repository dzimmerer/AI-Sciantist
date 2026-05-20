"""Tests for sciantist.wandb_ops module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from sciantist.wandb_ops import (
    _tail_text,
    _get_wandb_module,
    _parse_project_path,
    _find_run_by_name,
    _as_finite_float,
    _get_last_history_row_and_metric_histories,
    _download_log_files,
    _load_output_log_excerpt_from_payload,
    _to_json_safe,
    _build_payload,
    _extract_avg_gpu_metrics_from_events,
    _parse_project_path,
    create_tracking_backend,
)


class TestTailText:
    """Test _tail_text function."""

    def test_under_max_returns_same(self) -> None:
        text = "short text"
        result = _tail_text(text, 8000)
        assert result == text

    def test_over_max_returns_tail(self) -> None:
        text = "a" * 10000
        result = _tail_text(text, 100)
        assert len(result) == 100

    def test_exactly_max_returns_same(self) -> None:
        text = "x" * 100
        result = _tail_text(text, 100)
        assert result == text


class TestParseProjectPath:
    """Test _parse_project_path function."""

    def test_with_entity_format(self) -> None:
        result = _parse_project_path("my-project", "my-entity")
        assert result == "my-entity/my-project"

    def test_already_has_slash(self) -> None:
        result = _parse_project_path("entity/project", "other-entity")
        assert result == "entity/project"

    def test_uses_wandb_entity_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WANDB_ENTITY", "env-entity")
        result = _parse_project_path("my-project", "")
        assert result == "env-entity/my-project"

    def test_empty_entity_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WANDB_ENTITY", raising=False)
        with pytest.raises(RuntimeError, match="Entity is required"):
            _parse_project_path("my-project", "")


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

    def test_inf_returns_none(self) -> None:
        assert _as_finite_float(float("inf")) is None

    def test_nan_returns_none(self) -> None:
        assert _as_finite_float(float("nan")) is None


class TestToJsonSafe:
    """Test _to_json_safe function."""

    def test_primitives_unchanged(self) -> None:
        assert _to_json_safe(None) is None
        assert _to_json_safe("string") == "string"
        assert _to_json_safe(42) == 42
        assert _to_json_safe(3.14) == 3.14
        assert _to_json_safe(True) is True

    def test_dict_converts_keys(self) -> None:
        result = _to_json_safe({1: "one", 2: "two"})
        assert result == {"1": "one", "2": "two"}

    def test_list_unchanged(self) -> None:
        result = _to_json_safe([1, 2, "three"])
        assert result == [1, 2, "three"]

    def test_tuple_converts_to_list(self) -> None:
        result = _to_json_safe((1, 2, 3))
        assert result == [1, 2, 3]

    def test_set_converts_to_list(self) -> None:
        result = _to_json_safe({1, 2, 3})
        assert set(result) == {1, 2, 3}


class TestExtractAvgGpuMetricsFromEvents:
    """Test _extract_avg_gpu_metrics_from_events function."""

    def test_none_table_returns_nones(self) -> None:
        result = _extract_avg_gpu_metrics_from_events(None)
        assert result == (None, None)

    def test_empty_table_returns_nones(self) -> None:
        result = _extract_avg_gpu_metrics_from_events([])
        assert result == (None, None)

    def test_table_with_no_matching_metrics(self) -> None:
        rows = [{"other_metric": 0.5}, {"another": 0.8}]
        result = _extract_avg_gpu_metrics_from_events(rows)
        assert result == (None, None)

    def test_gpu_util_extracted(self) -> None:
        rows = [{"gpu_utilization": 50.0, "gpu_memory_percent": 70.0}]
        result = _extract_avg_gpu_metrics_from_events(rows)
        avg_util, avg_mem = result
        assert avg_util == 50.0

    def test_gpu_memory_percent_normalized_from_decimal(self) -> None:
        rows = [{"gpu_0_memory_utilization_percent": 0.5}]
        result = _extract_avg_gpu_metrics_from_events(rows)
        avg_util, avg_mem = result
        assert avg_mem == 50.0


class TestCreateTrackingBackend:
    """Test create_tracking_backend function."""

    def test_wandb_backend(self) -> None:
        from sciantist.wandb_ops import WandbTrackingBackend
        result = create_tracking_backend("wandb")
        assert isinstance(result, WandbTrackingBackend)

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported tracking backend"):
            create_tracking_backend("unknown")

    def test_case_insensitive(self) -> None:
        result = create_tracking_backend("WAndB")
        assert result.__class__.__name__ == "WandbTrackingBackend"


class TestLoadOutputLogExcerptFromPayload:
    """Test _load_output_log_excerpt_from_payload function."""

    def test_empty_downloaded_files(self) -> None:
        payload = {"downloaded_log_files": []}
        result = _load_output_log_excerpt_from_payload(payload)
        assert result == ""

    def test_non_list_downloaded_files(self) -> None:
        payload = {"downloaded_log_files": "not a list"}
        result = _load_output_log_excerpt_from_payload(payload)
        assert result == ""

    def test_reads_output_log(self, tmp_path: Path) -> None:
        log_path = tmp_path / "output.log"
        log_path.write_text("Log line 1\nLog line 2\nLog line 3")
        payload = {"downloaded_log_files": [str(log_path)]}
        result = _load_output_log_excerpt_from_payload(payload, max_chars=100)
        assert "Log line" in result

    def test_fallback_to_other_log_file(self, tmp_path: Path) -> None:
        log_path = tmp_path / "error.log"
        log_path.write_text("Error log content")
        payload = {"downloaded_log_files": [str(log_path)]}
        result = _load_output_log_excerpt_from_payload(payload)
        assert "Error log content" in result

    def test_max_chars_truncation(self, tmp_path: Path) -> None:
        log_path = tmp_path / "output.log"
        log_path.write_text("x" * 1000)
        payload = {"downloaded_log_files": [str(log_path)]}
        result = _load_output_log_excerpt_from_payload(payload, max_chars=100)
        assert len(result) <= 100