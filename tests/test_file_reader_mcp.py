"""Tests for llmclient.file_reader_mcp module."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from llmclient.file_reader_mcp import get_safe_path, list_directory, read_file


class TestGetSafePath:
    """Test get_safe_path function."""

    def test_resolves_relative_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MCP_BASE_DIR", "/tmp/base")
        import llmclient.file_reader_mcp as fr
        import importlib
        importlib.reload(fr)
        result = fr.get_safe_path("subdir/file.txt")
        assert result == Path("/tmp/base/subdir/file.txt")

    def test_denies_path_traversal_outside_base(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("MCP_BASE_DIR", str(tmp_path))
        import llmclient.file_reader_mcp as fr
        import importlib
        importlib.reload(fr)
        with pytest.raises(ValueError, match="Security Error"):
            fr.get_safe_path("../outside")

    def test_denies_path_traversal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MCP_BASE_DIR", "/tmp/base")
        import llmclient.file_reader_mcp as fr
        import importlib
        importlib.reload(fr)
        with pytest.raises(ValueError, match="Security Error"):
            fr.get_safe_path("../outside")

    def test_strips_leading_separator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MCP_BASE_DIR", "/tmp/base")
        import llmclient.file_reader_mcp as fr
        import importlib
        importlib.reload(fr)
        result = fr.get_safe_path("/subdir/file.txt")
        assert result == Path("/tmp/base/subdir/file.txt")

    def test_handles_empty_base_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MCP_BASE_DIR", raising=False)
        import llmclient.file_reader_mcp as fr
        import importlib
        importlib.reload(fr)
        result = fr.get_safe_path("file.txt")
        assert result.is_absolute()


class TestListDirectory:
    """Test list_directory function."""

    def test_invalid_directory_returns_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("MCP_BASE_DIR", str(tmp_path))
        import llmclient.file_reader_mcp as fr
        import importlib
        importlib.reload(fr)
        result = fr.list_directory("nonexistent")
        assert "Error" in result[0]

    def test_valid_directory_returns_files(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("MCP_BASE_DIR", str(tmp_path))
        import llmclient.file_reader_mcp as fr
        import importlib
        importlib.reload(fr)
        (tmp_path / "file.txt").touch()
        (tmp_path / "subdir").mkdir()
        result = fr.list_directory("")
        assert "file.txt" in result
        assert "subdir" in result

    def test_exception_returns_error_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MCP_BASE_DIR", "/nonexistent")
        import llmclient.file_reader_mcp as fr
        import importlib
        importlib.reload(fr)
        result = fr.list_directory("..")
        assert "Error" in result[0]


class TestReadFile:
    """Test read_file function."""

    def test_reads_file_content(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("MCP_BASE_DIR", str(tmp_path))
        import llmclient.file_reader_mcp as fr
        import importlib
        importlib.reload(fr)
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        result = fr.read_file("test.txt")
        assert result == "hello world"

    def test_missing_file_returns_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("MCP_BASE_DIR", str(tmp_path))
        import llmclient.file_reader_mcp as fr
        import importlib
        importlib.reload(fr)
        result = fr.read_file("nonexistent.txt")
        assert "Error" in result

    def test_invalid_path_returns_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("MCP_BASE_DIR", str(tmp_path))
        import llmclient.file_reader_mcp as fr
        import importlib
        importlib.reload(fr)
        result = fr.read_file("/etc/passwd")
        assert "Error" in result

    def test_exception_returns_error_string(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("MCP_BASE_DIR", str(tmp_path))
        import llmclient.file_reader_mcp as fr
        import importlib
        importlib.reload(fr)
        result = fr.read_file("../outside")
        assert "Error" in result