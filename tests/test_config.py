"""Tests for sciantist.config module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from sciantist.config import (
    ExperimentOutcome,
    LoopConfig,
    StageOutcome,
    _load_default_config_yaml,
    _load_repo_config_yaml,
)


class TestLoopConfig:
    """Test LoopConfig dataclass and its defaults."""

    def test_default_values(self) -> None:
        config = LoopConfig()
        assert config.target_repo == "/home/zimmerer/ws/chexclip/"
        assert config.branch_name == "autoresearch/loop"
        assert config.runtime == "05:00:00"
        assert config.poll_seconds == 60
        assert config.max_fix_attempts == 10
        assert config.tracking_backend == "wandb"
        assert config.wandb_project == "chexclip-sciantist"
        assert config.ideas_per_stage == 5
        assert config.async_worker_mode is True
        assert config.use_worktrees is True

    def test_metric_weights_default(self) -> None:
        config = LoopConfig()
        assert config.metric_weights == {
            "val/image_to_text_recall@10": 0.5,
            "val/text_to_image_recall@10": 0.5,
        }

    def test_custom_values(self) -> None:
        config = LoopConfig(
            target_repo="/custom/repo",
            runtime="02:00:00",
            wandb_project="custom/project",
            metric_higher_is_better=False,
        )
        assert config.target_repo == "/custom/repo"
        assert config.runtime == "02:00:00"
        assert config.wandb_project == "custom/project"
        assert config.metric_higher_is_better is False

    def test_allowed_file_suffixes_default(self) -> None:
        config = LoopConfig()
        assert config.allowed_file_suffixes == [".py", ".yaml", ".yml"]

    def test_denylist_patterns_not_empty(self) -> None:
        config = LoopConfig()
        assert len(config.denylist_patterns) > 0
        assert any(".venv" in p for p in config.denylist_patterns)

    def test_aider_only_patterns_default(self) -> None:
        config = LoopConfig()
        assert len(config.aider_only_patterns) > 0


class TestExperimentOutcome:
    """Test ExperimentOutcome dataclass."""

    def test_required_fields(self) -> None:
        outcome = ExperimentOutcome(
            timestamp_utc="2025-05-07T12:00:00Z",
            idea_title="Test idea",
            idea_branch_name="test-idea",
            feature_branch="feat/test-idea",
            idea_outline="Test outline",
            aider_plan_prompt="plan",
            aider_impl_prompt="impl",
            baseline_commit="abc123",
            trial_commit="def456",
            job_id="123",
            wandb_project="test/project",
            status="finished",
            runtime_seconds=3600,
            unified_metric=0.85,
            metric_histories={"val/x": [0.5, 0.85]},
            baseline_metric=0.80,
            metric_delta=0.05,
            avg_gpu_util=50.0,
            avg_gpu_memory=70.0,
            kept=True,
            summary="Test summary",
            info_log_excerpt="info log",
            wandb_log_excerpt="wandb log",
            err_log_excerpt="error log",
        )
        assert outcome.idea_title == "Test idea"
        assert outcome.status == "finished"
        assert outcome.unified_metric == 0.85
        assert outcome.kept is True
        assert outcome.run_id == ""
        assert outcome.worker_id == ""

    def test_optional_fields_defaults(self) -> None:
        outcome = ExperimentOutcome(
            timestamp_utc="2025-05-07T12:00:00Z",
            idea_title="Test idea",
            idea_branch_name="test-idea",
            feature_branch="feat/test-idea",
            idea_outline="Test outline",
            aider_plan_prompt="plan",
            aider_impl_prompt="impl",
            baseline_commit="abc123",
            trial_commit="def456",
            job_id="123",
            wandb_project="test/project",
            status="finished",
            runtime_seconds=3600,
            unified_metric=0.85,
            metric_histories=None,
            baseline_metric=None,
            metric_delta=None,
            avg_gpu_util=None,
            avg_gpu_memory=None,
            kept=False,
            summary="summary",
            info_log_excerpt="",
            wandb_log_excerpt="",
            err_log_excerpt="",
        )
        assert outcome.run_id == ""
        assert outcome.worker_id == ""
        assert outcome.worker_role == "general"
        assert outcome.parent_entry_ids == []
        assert outcome.currently_in_best_path is False

    def test_parent_entry_ids_mutable(self) -> None:
        outcome = ExperimentOutcome(
            timestamp_utc="2025-05-07T12:00:00Z",
            idea_title="Test idea",
            idea_branch_name="test-idea",
            feature_branch="feat/test-idea",
            idea_outline="Test outline",
            aider_plan_prompt="plan",
            aider_impl_prompt="impl",
            baseline_commit="abc123",
            trial_commit="def456",
            job_id="123",
            wandb_project="test/project",
            status="finished",
            runtime_seconds=None,
            unified_metric=None,
            metric_histories=None,
            baseline_metric=None,
            metric_delta=None,
            avg_gpu_util=None,
            avg_gpu_memory=None,
            kept=False,
            summary="summary",
            info_log_excerpt="",
            wandb_log_excerpt="",
            err_log_excerpt="",
            parent_entry_ids=["id1", "id2"],
        )
        assert outcome.parent_entry_ids == ["id1", "id2"]


class TestStageOutcome:
    """Test StageOutcome dataclass."""

    def test_fields(self) -> None:
        stage = StageOutcome(
            stage_index=1,
            baseline_commit="abc123",
            baseline_metric=0.80,
            merged_feature_branches=["feat/a", "feat/b"],
            skipped_conflict_feature_branches=["feat/c"],
            candidate_count=5,
            new_base_commit="xyz789",
            best_metric_after_stage=0.85,
            best_commit_after_stage="xyz789",
            stage_improvement_ideas="ideas here",
            summary="stage summary",
        )
        assert stage.stage_index == 1
        assert stage.baseline_metric == 0.80
        assert len(stage.merged_feature_branches) == 2


class TestLoadRepoConfigYaml:
    """Test _load_repo_config_yaml function."""

    def test_missing_file_returns_empty_dict(self) -> None:
        result = _load_repo_config_yaml("/nonexistent/path/config.yaml")
        assert result == {}

    def test_empty_yaml_returns_empty_dict(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()
            path = f.name
        try:
            result = _load_repo_config_yaml(path)
            assert result == {}
        finally:
            Path(path).unlink()

    def test_valid_yaml_returns_dict(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("target_repo: /test/repo\nbranch_name: test-branch\n")
            f.flush()
            path = f.name
        try:
            result = _load_repo_config_yaml(path)
            assert result == {"target_repo": "/test/repo", "branch_name": "test-branch"}
        finally:
            Path(path).unlink()

    def test_yaml_with_null_returns_empty_dict(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("\n")
            f.flush()
            path = f.name
        try:
            result = _load_repo_config_yaml(path)
            assert result == {}
        finally:
            Path(path).unlink()

    def test_yaml_with_list_raises_error(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("- item1\n- item2\n")
            f.flush()
            path = f.name
        try:
            with pytest.raises(ValueError, match="must be a YAML mapping"):
                _load_repo_config_yaml(path)
        finally:
            Path(path).unlink()


class TestLoadDefaultConfigYaml:
    """Test _load_default_config_yaml function."""

    def test_missing_file_returns_empty_dict(self) -> None:
        result = _load_default_config_yaml("/nonexistent/default_config.yaml")
        assert result == {}

    def test_empty_yaml_returns_empty_dict(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()
            path = f.name
        try:
            result = _load_default_config_yaml(path)
            assert result == {}
        finally:
            Path(path).unlink()

    def test_valid_yaml_returns_dict(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("default_branch: main\n")
            f.flush()
            path = f.name
        try:
            result = _load_default_config_yaml(path)
            assert result == {"default_branch": "main"}
        finally:
            Path(path).unlink()

    def test_yaml_with_list_raises_error(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("- item1\n- item2\n")
            f.flush()
            path = f.name
        try:
            with pytest.raises(ValueError, match="must be a YAML mapping"):
                _load_default_config_yaml(path)
        finally:
            Path(path).unlink()