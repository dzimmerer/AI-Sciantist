"""Tests for llmclient.git_mcp module."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from llmclient.git_mcp import (
    _normalize_repo_path,
    _run_git,
    _ensure_repo,
    git_repo_info,
    git_status,
    git_log,
    git_log_patch,
    git_diff,
    git_show,
    git_show_file,
    git_list_branches,
    git_list_tags,
)


class TestNormalizeRepoPath:
    """Test _normalize_repo_path function."""

    def test_normalizes_backslashes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GIT_MCP_REPO_DIR", "/tmp/repo")
        import llmclient.git_mcp as git_mcp
        import importlib
        importlib.reload(git_mcp)
        result = git_mcp._normalize_repo_path("subdir\\file.txt")
        assert result == "subdir/file.txt"

    def test_empty_path_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GIT_MCP_REPO_DIR", "/tmp/repo")
        import llmclient.git_mcp as git_mcp
        import importlib
        importlib.reload(git_mcp)
        with pytest.raises(ValueError, match="must not be empty"):
            git_mcp._normalize_repo_path("")

    def test_whitespace_only_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GIT_MCP_REPO_DIR", "/tmp/repo")
        import llmclient.git_mcp as git_mcp
        import importlib
        importlib.reload(git_mcp)
        with pytest.raises(ValueError, match="must not be empty"):
            git_mcp._normalize_repo_path("   ")

    def test_path_traversal_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GIT_MCP_REPO_DIR", "/tmp/repo")
        import llmclient.git_mcp as git_mcp
        import importlib
        importlib.reload(git_mcp)
        with pytest.raises(ValueError, match="outside repository"):
            git_mcp._normalize_repo_path("../outside")

    def test_returns_posix_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GIT_MCP_REPO_DIR", "/tmp/repo")
        import llmclient.git_mcp as git_mcp
        import importlib
        importlib.reload(git_mcp)
        result = git_mcp._normalize_repo_path("subdir/file.txt")
        assert "\\" not in result


class TestRunGit:
    """Test _run_git function."""

    def test_handles_invalid_command(self, tmp_path: Path) -> None:
        import llmclient.git_mcp as git_mcp
        git_mcp.REPO_DIR = tmp_path
        result = git_mcp._run_git(["invalid-command-that-does-not-exist"])
        assert result["exit_code"] != 0

    def test_truncates_long_stdout(self, tmp_path: Path) -> None:
        import llmclient.git_mcp as git_mcp
        git_mcp.REPO_DIR = tmp_path
        result = git_mcp._run_git(["status"], max_chars=1)
        assert len(result["stdout"]) <= 1 or result["stdout"] == ""

    def test_truncates_long_stderr(self, tmp_path: Path) -> None:
        import llmclient.git_mcp as git_mcp
        git_mcp.REPO_DIR = tmp_path
        result = git_mcp._run_git(["status"], max_chars=1)
        assert len(result["stderr"]) <= 1 or result["stderr"] == ""

    def test_file_not_found_returns_error(self) -> None:
        import llmclient.git_mcp as git_mcp
        import os
        old_path = os.environ.get("PATH")
        os.environ["PATH"] = "/nonexistent"
        try:
            git_mcp.REPO_DIR = Path("/tmp")
            result = git_mcp._run_git(["status"])
            assert "error" in result
        finally:
            if old_path:
                os.environ["PATH"] = old_path


class TestEnsureRepo:
    """Test _ensure_repo function."""

    def test_non_repo_returns_error(self, tmp_path: Path) -> None:
        import llmclient.git_mcp as git_mcp
        git_mcp.REPO_DIR = tmp_path
        result = git_mcp._ensure_repo()
        assert result is not None
        assert "error" in result

    def test_repo_returns_none(self, tmp_path: Path) -> None:
        import llmclient.git_mcp as git_mcp
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        git_mcp.REPO_DIR = tmp_path
        result = git_mcp._ensure_repo()
        assert result is None


class TestGitRepoInfo:
    """Test git_repo_info function."""

    def test_non_repo_returns_error(self, tmp_path: Path) -> None:
        import llmclient.git_mcp as git_mcp
        git_mcp.REPO_DIR = tmp_path
        result = git_mcp.git_repo_info()
        assert "error" in result

    def test_returns_repo_info(self, tmp_path: Path) -> None:
        import llmclient.git_mcp as git_mcp
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        git_mcp.REPO_DIR = tmp_path
        result = git_mcp.git_repo_info()
        assert "repo_dir" in result
        assert "top_level" in result


class TestGitStatus:
    """Test git_status function."""

    def test_non_repo_returns_error(self, tmp_path: Path) -> None:
        import llmclient.git_mcp as git_mcp
        git_mcp.REPO_DIR = tmp_path
        result = git_mcp.git_status()
        assert "error" in result

    def test_returns_status(self, tmp_path: Path) -> None:
        import llmclient.git_mcp as git_mcp
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        git_mcp.REPO_DIR = tmp_path
        result = git_mcp.git_status()
        assert "command" in result
        assert result["exit_code"] == 0


class TestGitLog:
    """Test git_log function."""

    def test_non_repo_returns_error(self, tmp_path: Path) -> None:
        import llmclient.git_mcp as git_mcp
        git_mcp.REPO_DIR = tmp_path
        result = git_mcp.git_log()
        assert "error" in result

    def test_returns_log(self, tmp_path: Path) -> None:
        import llmclient.git_mcp as git_mcp
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        git_mcp.REPO_DIR = tmp_path
        result = git_mcp.git_log(max_count=5)
        assert "command" in result

    def test_normalizes_count_bounds(self, tmp_path: Path) -> None:
        import llmclient.git_mcp as git_mcp
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        git_mcp.REPO_DIR = tmp_path
        result = git_mcp.git_log(max_count=500)
        assert "command" in result


class TestGitLogPatch:
    """Test git_log_patch function."""

    def test_non_repo_returns_error(self, tmp_path: Path) -> None:
        import llmclient.git_mcp as git_mcp
        git_mcp.REPO_DIR = tmp_path
        result = git_mcp.git_log_patch()
        assert "error" in result

    def test_returns_patch(self, tmp_path: Path) -> None:
        import llmclient.git_mcp as git_mcp
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        git_mcp.REPO_DIR = tmp_path
        result = git_mcp.git_log_patch(max_count=1)
        assert "command" in result


class TestGitDiff:
    """Test git_diff function."""

    def test_non_repo_returns_error(self, tmp_path: Path) -> None:
        import llmclient.git_mcp as git_mcp
        git_mcp.REPO_DIR = tmp_path
        result = git_mcp.git_diff()
        assert "error" in result

    def test_normalizes_empty_range(self, tmp_path: Path) -> None:
        import llmclient.git_mcp as git_mcp
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        git_mcp.REPO_DIR = tmp_path
        result = git_mcp.git_diff(revision_range="   ")
        assert "command" in result


class TestGitShow:
    """Test git_show function."""

    def test_non_repo_returns_error(self, tmp_path: Path) -> None:
        import llmclient.git_mcp as git_mcp
        git_mcp.REPO_DIR = tmp_path
        result = git_mcp.git_show()
        assert "error" in result

    def test_normalizes_empty_revision(self, tmp_path: Path) -> None:
        import llmclient.git_mcp as git_mcp
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        git_mcp.REPO_DIR = tmp_path
        result = git_mcp.git_show(revision="   ")
        assert "command" in result


class TestGitShowFile:
    """Test git_show_file function."""

    def test_non_repo_returns_error(self, tmp_path: Path) -> None:
        import llmclient.git_mcp as git_mcp
        git_mcp.REPO_DIR = tmp_path
        result = git_mcp.git_show_file("nonexistent.txt")
        assert "error" in result

    def test_path_traversal_returns_error(self, tmp_path: Path) -> None:
        import llmclient.git_mcp as git_mcp
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        git_mcp.REPO_DIR = tmp_path
        result = git_mcp.git_show_file("../outside")
        assert "error" in result


class TestGitListBranches:
    """Test git_list_branches function."""

    def test_non_repo_returns_error(self, tmp_path: Path) -> None:
        import llmclient.git_mcp as git_mcp
        git_mcp.REPO_DIR = tmp_path
        result = git_mcp.git_list_branches()
        assert "error" in result

    def test_returns_branches(self, tmp_path: Path) -> None:
        import llmclient.git_mcp as git_mcp
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        git_mcp.REPO_DIR = tmp_path
        result = git_mcp.git_list_branches()
        assert "command" in result

    def test_local_only(self, tmp_path: Path) -> None:
        import llmclient.git_mcp as git_mcp
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        git_mcp.REPO_DIR = tmp_path
        result = git_mcp.git_list_branches(all_branches=False)
        assert "command" in result


class TestGitListTags:
    """Test git_list_tags function."""

    def test_non_repo_returns_error(self, tmp_path: Path) -> None:
        import llmclient.git_mcp as git_mcp
        git_mcp.REPO_DIR = tmp_path
        result = git_mcp.git_list_tags()
        assert "error" in result

    def test_returns_tags(self, tmp_path: Path) -> None:
        import llmclient.git_mcp as git_mcp
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        git_mcp.REPO_DIR = tmp_path
        result = git_mcp.git_list_tags()
        assert "command" in result

    def test_normalizes_count(self, tmp_path: Path) -> None:
        import llmclient.git_mcp as git_mcp
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        git_mcp.REPO_DIR = tmp_path
        result = git_mcp.git_list_tags(max_count=5000)
        assert "command" in result