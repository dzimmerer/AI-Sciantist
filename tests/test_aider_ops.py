"""Tests for sciantist.aider_ops module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sciantist.aider_ops import (
    _tail_text,
    _find_traceback_excerpt,
    _configure_llm_env,
    _detect_repo_root,
)


class TestTailText:
    """Test _tail_text function."""

    def test_under_max_returns_same(self) -> None:
        text = "hello world"
        result = _tail_text(text, max_chars=100)
        assert result == text

    def test_over_max_returns_tail(self) -> None:
        text = "a" * 100
        result = _tail_text(text, max_chars=10)
        assert result == "a" * 10

    def test_exactly_max_returns_same(self) -> None:
        text = "a" * 100
        result = _tail_text(text, max_chars=100)
        assert result == text


class TestFindTracebackExcerpt:
    """Test _find_traceback_excerpt function."""

    def test_no_traceback_returns_tail(self) -> None:
        text = "some log text without traceback"
        result = _find_traceback_excerpt(text)
        assert result == text[:3000]

    def test_traceback_found_returns_from_marker(self) -> None:
        text = "some log text\nTraceback (most recent call last):\n  File 'test.py', line 10\n    raise ValueError()\nValueError: test error"
        result = _find_traceback_excerpt(text)
        assert "Traceback (most recent call last):" in result

    def test_traceback_at_start(self) -> None:
        text = "Traceback (most recent call last):\n  File 'test.py', line 10\n    raise ValueError()"
        result = _find_traceback_excerpt(text)
        assert result == text

    def test_multiple_tracebacks_uses_last(self) -> None:
        text = "first traceback\nTraceback (most recent call last):\nfirst error\nsecond traceback\nTraceback (most recent call last):\nsecond error"
        result = _find_traceback_excerpt(text)
        assert "second error" in result

    def test_truncates_long_traceback(self) -> None:
        text = "some log\nTraceback (most recent call last):\n" + "x" * 10000
        result = _find_traceback_excerpt(text)
        assert len(result) <= 5000


class TestConfigureLlmEnv:
    """Test _configure_llm_env function."""

    def test_raises_for_empty_model(self) -> None:
        from sciantist.aider_ops import _configure_llm_env
        from sciantist.config import LoopConfig
        config = LoopConfig()
        config.aider_model = ""
        with pytest.raises(RuntimeError, match="Unsupported"):
            _configure_llm_env(config, "fake-key")

    def test_raises_for_unknown_provider(self) -> None:
        from sciantist.aider_ops import _configure_llm_env
        from sciantist.config import LoopConfig
        config = LoopConfig()
        config.aider_model = "unknown/model"
        config.openai_api_base = "https://api.example.com"
        with pytest.raises(RuntimeError, match="Unsupported"):
            _configure_llm_env(config, "fake-key")

    def test_sets_openai_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import os
        from sciantist.aider_ops import _configure_llm_env
        from sciantist.config import LoopConfig
        monkeypatch.delenv("OPENAI_API_BASE", raising=False)
        config = LoopConfig()
        config.aider_model = "openai/gpt-4"
        config.openai_api_base = "https://api.example.com/"
        _configure_llm_env(config, "fake-key")
        assert os.environ.get("OPENAI_API_BASE") == "https://api.example.com/v1"

    def test_sets_anthropic_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import os
        from sciantist.aider_ops import _configure_llm_env
        from sciantist.config import LoopConfig
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config = LoopConfig()
        config.aider_model = "anthropic/claude-3"
        config.openai_api_base = "https://api.example.com"
        _configure_llm_env(config, "fake-key")
        assert os.environ.get("ANTHROPIC_API_KEY") == "fake-key"


class TestDetectRepoRoot:
    """Test _detect_repo_root function."""

    def test_empty_files_returns_fallback(self) -> None:
        result = _detect_repo_root([], "/tmp/nonexistent")
        assert result == Path("/tmp/nonexistent")

    def test_finds_repo_from_file_parent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import subprocess
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)

        test_file = tmp_path / "test.py"
        test_file.touch()

        result = _detect_repo_root([str(test_file)], "/tmp/fallback")
        assert result == tmp_path

    def test_skips_invalid_directories(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = _detect_repo_root([str(empty_dir)], str(tmp_path / "fallback"))
        assert result == Path(tmp_path / "fallback")

    def test_returns_fallback_when_no_git_repo(self, tmp_path: Path) -> None:
        result = _detect_repo_root([str(tmp_path / "test.py")], str(tmp_path))
        assert result == tmp_path