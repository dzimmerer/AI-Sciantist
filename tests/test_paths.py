"""Tests for sciantist.paths module."""

from __future__ import annotations

from pathlib import Path

import pytest

from sciantist.config import LoopConfig
from sciantist.paths import (
    _build_leaderboard_lock_path,
    _build_memory_lock_path,
    _build_repo_lock_path,
    _build_worker_memory_path,
    _build_worker_output_dir,
    _build_worker_run_id,
    _build_worker_worktree_path,
)


class TestBuildWorkerRunId:
    """Test _build_worker_run_id function."""

    def test_format_includes_worker_id(self) -> None:
        result = _build_worker_run_id("worker_01", 5)
        assert result.startswith("worker_01-r")

    def test_format_includes_run_index(self) -> None:
        result = _build_worker_run_id("worker_01", 5)
        assert "-r000005-" in result

    def test_format_has_uuid_suffix(self) -> None:
        result = _build_worker_run_id("worker_01", 5)
        parts = result.split("-")
        assert len(parts) >= 3


class TestBuildWorkerWorktreePath:
    """Test _build_worker_worktree_path function."""

    def test_includes_worker_id(self) -> None:
        result = _build_worker_worktree_path(Path("/output"), "worker_01", 5)
        assert "worker_01" in str(result)

    def test_includes_run_index_padded(self) -> None:
        result = _build_worker_worktree_path(Path("/output"), "worker_01", 5)
        assert "run_000005" in str(result)

    def test_path_structure(self) -> None:
        result = _build_worker_worktree_path(Path("/output"), "worker_01", 5)
        assert "worktrees/worker_01/run_000005" in str(result)


class TestBuildWorkerOutputDir:
    """Test _build_worker_output_dir function."""

    def test_includes_worker_id(self) -> None:
        result = _build_worker_output_dir(Path("/output"), "worker_01", 5)
        assert "worker_01" in str(result)

    def test_includes_run_index_padded(self) -> None:
        result = _build_worker_output_dir(Path("/output"), "worker_01", 5)
        assert "run_000005" in str(result)

    def test_path_structure(self) -> None:
        result = _build_worker_output_dir(Path("/output"), "worker_01", 5)
        assert "workers/worker_01/run_000005" in str(result)


class TestBuildWorkerMemoryPath:
    """Test _build_worker_memory_path function."""

    def test_includes_worker_id(self) -> None:
        result = _build_worker_memory_path(Path("/output"), "worker_01")
        assert "worker_01" in str(result)

    def test_memory_filename(self) -> None:
        result = _build_worker_memory_path(Path("/output"), "worker_01")
        assert result.name == "memory.md"

    def test_path_structure(self) -> None:
        result = _build_worker_memory_path(Path("/output"), "worker_01")
        assert "workers/worker_01/memory.md" in str(result)


class TestBuildLeaderboardLockPath:
    """Test _build_leaderboard_lock_path function."""

    def test_uses_config_lock_file_name(self) -> None:
        config = LoopConfig(lock_file_name="custom.lock")
        result = _build_leaderboard_lock_path(Path("/output"), config)
        assert result.name == "custom.lock"

    def test_default_lock_file_name(self) -> None:
        config = LoopConfig()
        result = _build_leaderboard_lock_path(Path("/output"), config)
        assert result.name == ".sciantist.lock"


class TestBuildRepoLockPath:
    """Test _build_repo_lock_path function."""

    def test_lock_file_name(self) -> None:
        result = _build_repo_lock_path(Path("/output"))
        assert result.name == "repo.lock"

    def test_path_is_under_output_root(self) -> None:
        result = _build_repo_lock_path(Path("/output"))
        assert result.parent == Path("/output")


class TestBuildMemoryLockPath:
    """Test _build_memory_lock_path function."""

    def test_lock_file_name(self) -> None:
        result = _build_memory_lock_path(Path("/output"))
        assert result.name == "memory.lock"

    def test_path_is_under_output_root(self) -> None:
        result = _build_memory_lock_path(Path("/output"))
        assert result.parent == Path("/output")