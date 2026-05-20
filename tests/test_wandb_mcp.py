"""Tests for llmclient.wandb_mcp module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestNormalizeProjectPath:
    """Test _normalize_project_path function."""

    def test_empty_project_raises(self) -> None:
        import llmclient.wandb_mcp as wandb_mcp
        with pytest.raises(ValueError, match="must not be empty"):
            wandb_mcp._normalize_project_path("")

    def test_project_with_slash_returned_as_is(self) -> None:
        import llmclient.wandb_mcp as wandb_mcp
        result = wandb_mcp._normalize_project_path("entity/project")
        assert result == "entity/project"

    def test_project_without_entity_uses_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WANDB_ENTITY", "test_entity")
        import llmclient.wandb_mcp as wandb_mcp
        import importlib
        importlib.reload(wandb_mcp)
        result = wandb_mcp._normalize_project_path("my_project")
        assert result == "test_entity/my_project"

    def test_project_without_entity_and_no_env_raises(self) -> None:
        import llmclient.wandb_mcp as wandb_mcp
        import os
        old_entity = os.environ.pop("WANDB_ENTITY", None)
        try:
            import importlib
            importlib.reload(wandb_mcp)
            with pytest.raises(ValueError, match="entity must be provided"):
                wandb_mcp._normalize_project_path("my_project")
        finally:
            if old_entity:
                os.environ["WANDB_ENTITY"] = old_entity

    def test_entity_argument_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WANDB_ENTITY", "env_entity")
        import llmclient.wandb_mcp as wandb_mcp
        import importlib
        importlib.reload(wandb_mcp)
        result = wandb_mcp._normalize_project_path("my_project", entity="arg_entity")
        assert result == "arg_entity/my_project"


class TestToJsonSafe:
    """Test _to_json_safe function."""

    def test_primitives_unchanged(self) -> None:
        import llmclient.wandb_mcp as wandb_mcp
        assert wandb_mcp._to_json_safe(None) is None
        assert wandb_mcp._to_json_safe("test") == "test"
        assert wandb_mcp._to_json_safe(42) == 42
        assert wandb_mcp._to_json_safe(3.14) == 3.14
        assert wandb_mcp._to_json_safe(True) is True

    def test_dict_converts_keys_to_strings(self) -> None:
        import llmclient.wandb_mcp as wandb_mcp
        result = wandb_mcp._to_json_safe({1: "value", "key": 42})
        assert "1" in result
        assert result["1"] == "value"
        assert result["key"] == 42

    def test_list_tuple_set_converted(self) -> None:
        import llmclient.wandb_mcp as wandb_mcp
        assert wandb_mcp._to_json_safe([1, 2, 3]) == [1, 2, 3]
        assert wandb_mcp._to_json_safe((1, 2)) == [1, 2]
        assert wandb_mcp._to_json_safe({1, 2}) == [1, 2]

    def test_object_with_dict_attribute(self) -> None:
        import llmclient.wandb_mcp as wandb_mcp
        mock_obj = MagicMock()
        mock_obj._dict = {"key": "value"}
        result = wandb_mcp._to_json_safe(mock_obj)
        assert result == {"key": "value"}

    def test_object_with_to_dict_method(self) -> None:
        import llmclient.wandb_mcp as wandb_mcp

        class MockWithDict:
            def to_dict(self):
                return {"key": "value"}

        result = wandb_mcp._to_json_safe(MockWithDict())
        assert result == {"key": "value"}

    def test_fallback_to_str(self) -> None:
        import llmclient.wandb_mcp as wandb_mcp
        mock_obj = MagicMock()
        result = wandb_mcp._to_json_safe(mock_obj)
        assert isinstance(result, str)


class TestGetWandbModule:
    """Test _get_wandb_module function."""

    def test_wandb_not_installed_raises(self) -> None:
        import llmclient.wandb_mcp as wandb_mcp
        import importlib
        with patch("importlib.import_module", side_effect=ImportError("No module named 'wandb'")):
            with pytest.raises(RuntimeError, match="wandb"):
                wandb_mcp._get_wandb_module()


class TestGetRunInfo:
    """Test get_run_info function."""

    def test_error_returns_error_dict(self) -> None:
        import llmclient.wandb_mcp as wandb_mcp
        with patch.object(wandb_mcp, "_get_api_and_run", side_effect=Exception("API Error")):
            result = wandb_mcp.get_run_info("project", "run_id")
            assert "error" in result


class TestGetRunHistory:
    """Test get_run_history function."""

    def test_error_returns_error_dict(self) -> None:
        import llmclient.wandb_mcp as wandb_mcp
        with patch.object(wandb_mcp, "_get_api_and_run", side_effect=Exception("API Error")):
            result = wandb_mcp.get_run_history("project", "run_id")
            assert "error" in result

    def test_filters_metrics(self) -> None:
        import llmclient.wandb_mcp as wandb_mcp
        mock_run = MagicMock()
        mock_run.scan_history.return_value = iter([{"loss": 0.5}, {"loss": 0.3}])
        with patch.object(wandb_mcp, "_get_api_and_run", return_value=(None, mock_run, "p/r")):
            result = wandb_mcp.get_run_history("project", "run_id", metric_names=["loss"])
            assert result["row_count"] == 2


class TestGetRunMetricSeries:
    """Test get_run_metric_series function."""

    def test_empty_metric_names_returns_error(self) -> None:
        import llmclient.wandb_mcp as wandb_mcp
        result = wandb_mcp.get_run_metric_series("project", "run_id", metric_names=[])
        assert "error" in result

    def test_error_returns_error_dict(self) -> None:
        import llmclient.wandb_mcp as wandb_mcp
        with patch.object(wandb_mcp, "_get_api_and_run", side_effect=Exception("API Error")):
            result = wandb_mcp.get_run_metric_series("project", "run_id", metric_names=["loss"])
            assert "error" in result

    def test_handles_bool_values(self) -> None:
        import llmclient.wandb_mcp as wandb_mcp
        mock_run = MagicMock()
        mock_run.scan_history.return_value = iter([{"loss": 0.5, "is_train": True}])
        with patch.object(wandb_mcp, "_get_api_and_run", return_value=(None, mock_run, "p/r")):
            result = wandb_mcp.get_run_metric_series("project", "run_id", metric_names=["loss", "is_train"])
            assert "loss" in result["series"]


class TestListRunFiles:
    """Test list_run_files function."""

    def test_error_returns_error_dict(self) -> None:
        import llmclient.wandb_mcp as wandb_mcp
        with patch.object(wandb_mcp, "_get_api_and_run", side_effect=Exception("API Error")):
            result = wandb_mcp.list_run_files("project", "run_id")
            assert "error" in result


class TestDownloadRunFiles:
    """Test download_run_files function."""

    def test_error_returns_error_dict(self) -> None:
        import llmclient.wandb_mcp as wandb_mcp
        with patch.object(wandb_mcp, "_get_api_and_run", side_effect=Exception("API Error")):
            result = wandb_mcp.download_run_files("project", "run_id")
            assert "error" in result


class TestGetRunLogs:
    """Test get_run_logs function."""

    def test_error_from_download_returns_error(self) -> None:
        import llmclient.wandb_mcp as wandb_mcp
        with patch.object(wandb_mcp, "download_run_files", return_value={"error": "Download failed"}):
            result = wandb_mcp.get_run_logs("project", "run_id")
            assert "error" in result


class TestGetRunFullSnapshot:
    """Test get_run_full_snapshot function."""

    def test_error_from_get_run_info_returns_error(self) -> None:
        import llmclient.wandb_mcp as wandb_mcp
        with patch.object(wandb_mcp, "get_run_info", return_value={"error": "Run not found"}):
            result = wandb_mcp.get_run_full_snapshot("project", "run_id")
            assert "error" in result

    def test_snapshot_saves_to_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import llmclient.wandb_mcp as wandb_mcp
        monkeypatch.setenv("WANDB_MCP_DOWNLOAD_DIR", str(tmp_path))
        import importlib
        importlib.reload(wandb_mcp)

        mock_run = MagicMock()
        mock_run.entity = "entity"
        mock_run.project = "project"
        mock_run.id = "run_id"
        mock_run.scan_history.return_value = iter([])
        mock_run.files.return_value = iter([])

        with patch.object(wandb_mcp, "_get_api_and_run", return_value=(None, mock_run, "entity/project")):
            with patch.object(wandb_mcp, "get_run_info", return_value={
                "entity": "entity",
                "project": "project",
                "run_id": "run_id"
            }):
                with patch.object(wandb_mcp, "download_run_files", return_value={
                    "downloaded_files": [],
                    "project_path": "entity/project",
                    "run_id": "run_id"
                }):
                    result = wandb_mcp.get_run_full_snapshot("project", "run_id")
                    assert "snapshot_path" in result