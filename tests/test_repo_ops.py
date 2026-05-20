"""Tests for sciantist.repo_ops module."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from sciantist.repo_ops import (
    _run_command,
    _git,
    _git_try,
    _commit_tree_id,
    _push_to_origin,
    _resolve_train_command_for_repo,
    _candidate_worktree_path,
    _sanitize_branch_token,
    _make_feature_branch_name,
    _extract_diff_summary,
    _extract_diff_patch,
    _commit_all,
    _commit_if_dirty,
)


class TestRunCommand:
    """Test _run_command function."""

    def test_run_echo(self) -> None:
        result = _run_command(["echo", "hello"], cwd="/tmp")
        assert result.stdout.strip() == "hello"

    def test_run_invalid_command_raises(self) -> None:
        with pytest.raises(subprocess.CalledProcessError):
            _run_command(["false"], cwd="/tmp")


class TestGit:
    """Test _git function."""

    @patch("subprocess.run")
    def test_git_rev_parse(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(stdout="abc123\n", stderr="")
        result = _git("/tmp/repo", "rev-parse", "HEAD")
        assert result == "abc123"

    @patch("subprocess.run")
    def test_git_strips_whitespace(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(stdout="  abc123  \n", stderr="")
        result = _git("/tmp/repo", "rev-parse", "HEAD")
        assert result == "abc123"


class TestGitTry:
    """Test _git_try function."""

    @patch("sciantist.repo_ops._git")
    def test_success_returns_true(self, mock_git: MagicMock) -> None:
        mock_git.return_value = "output"
        success, output = _git_try("/tmp/repo", "status")
        assert success is True
        assert output == "output"

    @patch("sciantist.repo_ops._git")
    def test_failure_returns_false(self, mock_git: MagicMock) -> None:
        mock_git.side_effect = subprocess.CalledProcessError(1, "git", stderr="error")
        success, output = _git_try("/tmp/repo", "status")
        assert success is False
        assert "error" in output


class TestCommitTreeId:
    """Test _commit_tree_id function."""

    @patch("sciantist.repo_ops._git_try")
    def test_valid_commit(self, mock_git_try: MagicMock) -> None:
        mock_git_try.return_value = (True, "tree123")
        result = _commit_tree_id("/tmp/repo", "abc123")
        assert result == "tree123"

    @patch("sciantist.repo_ops._git_try")
    def test_invalid_commit_returns_empty(self, mock_git_try: MagicMock) -> None:
        mock_git_try.return_value = (False, "not a commit")
        result = _commit_tree_id("/tmp/repo", "invalid")
        assert result == ""

    def test_empty_commit_returns_empty(self) -> None:
        result = _commit_tree_id("/tmp/repo", "")
        assert result == ""


class TestPushToOrigin:
    """Test _push_to_origin function."""

    @patch("sciantist.repo_ops._run_command")
    def test_push_all_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(stdout="Everything up-to-date", stderr="")
        success, output = _push_to_origin("/tmp/repo")
        assert success is True

    @patch("sciantist.repo_ops._run_command")
    def test_push_ref_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(stdout="To github.com:...", stderr="")
        success, output = _push_to_origin("/tmp/repo", ref="feature-branch")
        assert success is True

    @patch("sciantist.repo_ops._run_command")
    def test_push_failure(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.CalledProcessError(1, "git push", stderr="error")
        success, output = _push_to_origin("/tmp/repo")
        assert success is False
        assert "error" in output


class TestResolveTrainCommandForRepo:
    """Test _resolve_train_command_for_repo function."""

    def test_relative_first_token_prefixed(self) -> None:
        result = _resolve_train_command_for_repo(
            "python scripts/train.py",
            "/source/repo",
            "/target/repo",
        )
        assert result.startswith("/target/repo")

    def test_absolute_matching_source_replaced(self) -> None:
        result = _resolve_train_command_for_repo(
            "/source/repo/scripts/train.py",
            "/source/repo",
            "/target/repo",
        )
        assert "/source/repo" not in result
        assert "/target/repo" in result

    def test_absolute_different_source_unchanged(self) -> None:
        result = _resolve_train_command_for_repo(
            "/other/repo/scripts/train.py",
            "/source/repo",
            "/target/repo",
        )
        assert "/other/repo" in result

    def test_empty_command_returns_empty(self) -> None:
        result = _resolve_train_command_for_repo("", "/source", "/target")
        assert result == ""

    def test_invalid_shlex_parsing_returns_original(self) -> None:
        result = _resolve_train_command_for_repo("command with unbalanced '", "/source", "/target")
        assert result == "command with unbalanced '"


class TestCandidateWorktreePath:
    """Test _candidate_worktree_path function."""

    def test_format(self) -> None:
        result = _candidate_worktree_path(Path("/output"), 1, 5)
        assert "stage_0001" in str(result)
        assert "candidate_05" in str(result)

    def test_path_structure(self) -> None:
        result = _candidate_worktree_path(Path("/output"), 1, 5)
        assert "worktrees" in str(result)


class TestSanitizeBranchToken:
    """Test _sanitize_branch_token function."""

    def test_replaces_special_chars(self) -> None:
        result = _sanitize_branch_token("test@#$idea")
        assert result == "test-idea"

    def test_trims_whitespace(self) -> None:
        result = _sanitize_branch_token("  test idea  ")
        assert result == "test-idea"

    def test_lowercases(self) -> None:
        result = _sanitize_branch_token("TestIdea")
        assert result == "testidea"

    def test_empty_becomes_idea(self) -> None:
        result = _sanitize_branch_token("   ")
        assert result == "idea"

    def test_truncates_to_40_chars(self) -> None:
        result = _sanitize_branch_token("x" * 100)
        assert len(result) == 40

    def test_multiple_dashes_preserved(self) -> None:
        result = _sanitize_branch_token("test---idea")
        assert result == "test---idea"


class TestExtractDiffPatch:
    """Test _extract_diff_patch function."""

    @patch("sciantist.repo_ops._git_try")
    def test_truncates_long_patch(self, mock_git_try: MagicMock) -> None:
        mock_git_try.return_value = (True, "x" * 50000)
        result = _extract_diff_patch("/repo", "abc", "def", max_chars=1000)
        assert "..." in result


class TestCommitAll:
    """Test _commit_all function."""

    @patch("sciantist.repo_ops._git")
    def test_returns_commit_hash(self, mock_git: MagicMock) -> None:
        mock_git.side_effect = ["", "", "abc123"]
        result = _commit_all("/repo", "test message")
        assert result == "abc123"


class TestCommitIfDirty:
    """Test _commit_if_dirty function."""

    @patch("sciantist.repo_ops._git")
    def test_not_dirty_returns_none(self, mock_git: MagicMock) -> None:
        mock_git.return_value = ""
        result = _commit_if_dirty("/repo", "message")
        assert result is None

    @patch("sciantist.repo_ops._git")
    def test_dirty_returns_commit_hash(self, mock_git: MagicMock) -> None:
        mock_git.side_effect = ["M file.py", "", "", "def456"]
        result = _commit_if_dirty("/repo", "message")
        assert result == "def456"