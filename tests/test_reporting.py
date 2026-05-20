"""Tests for sciantist.reporting module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from sciantist.config import ExperimentOutcome
from sciantist.reporting import (
    _tail_text,
    _format_runtime_dd_hh_mm_ss,
    _candidate_label_from_feature_branch,
    _summarize_metric_histories,
    _render_metric_histories_markdown_lines,
    _render_experiment_outcome_markdown,
    _write_stage_candidate_summary_markdown,
    _write_stage_summary_markdown,
    _append_experiments_md,
    _append_experiments_tsv,
    _is_outcome_already_logged,
)


class TestReportingTailText:
    """Test _tail_text function from reporting module."""

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


class TestFormatRuntimeDdHhMmSs:
    """Test _format_runtime_dd_hh_mm_ss function."""

    def test_none_returns_unknown(self) -> None:
        result = _format_runtime_dd_hh_mm_ss(None)
        assert result == "(unknown)"

    def test_zero(self) -> None:
        result = _format_runtime_dd_hh_mm_ss(0)
        assert result == "00-00:00:00"

    def test_seconds_only(self) -> None:
        result = _format_runtime_dd_hh_mm_ss(45)
        assert result == "00-00:00:45"

    def test_hours_minutes_seconds(self) -> None:
        result = _format_runtime_dd_hh_mm_ss(3661)
        assert result == "00-01:01:01"

    def test_days_hours_minutes_seconds(self) -> None:
        result = _format_runtime_dd_hh_mm_ss(90061)
        assert result == "01-01:01:01"

    def test_negative_becomes_zero(self) -> None:
        result = _format_runtime_dd_hh_mm_ss(-100)
        assert result == "00-00:00:00"


class TestCandidateLabelFromFeatureBranch:
    """Test _candidate_label_from_feature_branch function."""

    def test_baseline(self) -> None:
        result = _candidate_label_from_feature_branch("stage-baseline-abc123")
        assert result == "baseline"

    def test_candidate(self) -> None:
        result = _candidate_label_from_feature_branch("feat/test_c01_some_branch")
        assert result == "candidate_01"

    def test_unknown_when_no_marker(self) -> None:
        result = _candidate_label_from_feature_branch("feat/test-branch")
        assert result == "unknown"

    def test_unknown_when_no_digits(self) -> None:
        result = _candidate_label_from_feature_branch("feat/test_c_branch")
        assert result == "unknown"


class TestSummarizeMetricHistories:
    """Test _summarize_metric_histories function."""

    def test_none_returns_no_histories(self) -> None:
        result = _summarize_metric_histories(None)
        assert result == "(no metric histories)"

    def test_empty_dict_returns_no_histories(self) -> None:
        result = _summarize_metric_histories({})
        assert result == "(no metric histories)"

    def test_single_metric(self) -> None:
        histories = {"val/accuracy": [0.5, 0.7, 0.9]}
        result = _summarize_metric_histories(histories)
        assert "val/accuracy" in result
        assert "steps=3" in result

    def test_multiple_metrics(self) -> None:
        histories = {
            "val/accuracy": [0.5, 0.7, 0.9],
            "val/loss": [2.0, 1.5, 1.0],
        }
        result = _summarize_metric_histories(histories)
        assert "val/accuracy" in result
        assert "val/loss" in result

    def test_empty_list_values(self) -> None:
        histories = {"val/empty": []}
        result = _summarize_metric_histories(histories)
        assert result == "(no metric histories)"


class TestRenderMetricHistoriesMarkdownLines:
    """Test _render_metric_histories_markdown_lines function."""

    def test_none_returns_no_histories(self) -> None:
        result = _render_metric_histories_markdown_lines(None)
        assert "(no metric histories)" in result

    def test_empty_dict_returns_no_histories(self) -> None:
        result = _render_metric_histories_markdown_lines({})
        assert "(no metric histories)" in result

    def test_renders_metric_values(self) -> None:
        histories = {"val/accuracy": [0.5, 0.7, 0.9]}
        result = _render_metric_histories_markdown_lines(histories)
        result_str = "\n".join(result)
        assert "val/accuracy" in result_str
        assert "steps: 3" in result_str
        assert "[0.500000, 0.700000, 0.900000]" in result_str


class TestRenderExperimentOutcomeMarkdown:
    """Test _render_experiment_outcome_markdown function."""

    def test_renders_basic_fields(self) -> None:
        outcome = ExperimentOutcome(
            timestamp_utc="2025-05-07T12:00:00Z",
            idea_title="Test Idea",
            idea_branch_name="test-idea",
            feature_branch="feat/test-idea_c01",
            idea_outline="Test outline",
            aider_plan_prompt="plan prompt",
            aider_impl_prompt="impl prompt",
            baseline_commit="abc123",
            trial_commit="def456",
            job_id="12345",
            wandb_project="test/project",
            status="finished",
            runtime_seconds=3600,
            unified_metric=0.85,
            metric_histories={"val/acc": [0.5, 0.85]},
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
        result = _render_experiment_outcome_markdown(outcome, heading_level=2, include_logs=False)
        assert "Test Idea" in result
        assert "test-idea" in result
        assert "finished" in result
        assert "0.85" in result
        assert "candidate_01" in result

    def test_include_logs_true(self) -> None:
        outcome = ExperimentOutcome(
            timestamp_utc="2025-05-07T12:00:00Z",
            idea_title="Test",
            idea_branch_name="test",
            feature_branch="feat/test_c01",
            idea_outline="outline",
            aider_plan_prompt="plan",
            aider_impl_prompt="impl",
            baseline_commit="abc",
            trial_commit="def",
            job_id="123",
            wandb_project="test/project",
            status="finished",
            runtime_seconds=100,
            unified_metric=0.5,
            metric_histories=None,
            baseline_metric=0.4,
            metric_delta=0.1,
            avg_gpu_util=None,
            avg_gpu_memory=None,
            kept=False,
            summary="summary",
            info_log_excerpt="info",
            wandb_log_excerpt="wandb",
            err_log_excerpt="err",
        )
        result = _render_experiment_outcome_markdown(outcome, include_logs=True)
        assert "Error Log Excerpt" in result
        assert "Info Log Excerpt" in result

    def test_heading_level(self) -> None:
        outcome = ExperimentOutcome(
            timestamp_utc="2025-05-07T12:00:00Z",
            idea_title="Test",
            idea_branch_name="test",
            feature_branch="feat/test_c01",
            idea_outline="outline",
            aider_plan_prompt="plan",
            aider_impl_prompt="impl",
            baseline_commit="abc",
            trial_commit="def",
            job_id="123",
            wandb_project="test/project",
            status="finished",
            runtime_seconds=100,
            unified_metric=0.5,
            metric_histories=None,
            baseline_metric=0.4,
            metric_delta=0.1,
            avg_gpu_util=None,
            avg_gpu_memory=None,
            kept=False,
            summary="summary",
            info_log_excerpt="",
            wandb_log_excerpt="",
            err_log_excerpt="",
        )
        result = _render_experiment_outcome_markdown(outcome, heading_level=3)
        assert result.startswith("### ")


class TestWriteStageCandidateSummaryMarkdown:
    """Test _write_stage_candidate_summary_markdown function."""

    def test_creates_candidate_directory(self, tmp_path: Path) -> None:
        stage_dir = tmp_path / "stage_01"
        outcome = ExperimentOutcome(
            timestamp_utc="2025-05-07T12:00:00Z",
            idea_title="Test",
            idea_branch_name="test",
            feature_branch="feat/test_c01",
            idea_outline="outline",
            aider_plan_prompt="plan",
            aider_impl_prompt="impl",
            baseline_commit="abc",
            trial_commit="def",
            job_id="123",
            wandb_project="test/project",
            status="finished",
            runtime_seconds=100,
            unified_metric=0.5,
            metric_histories=None,
            baseline_metric=0.4,
            metric_delta=0.1,
            avg_gpu_util=None,
            avg_gpu_memory=None,
            kept=False,
            summary="summary",
            info_log_excerpt="",
            wandb_log_excerpt="",
            err_log_excerpt="",
        )
        path = _write_stage_candidate_summary_markdown(stage_dir, "candidate_01", outcome)
        assert path.exists()
        assert "candidate_01" in str(path)


class TestWriteStageSummaryMarkdown:
    """Test _write_stage_summary_markdown function."""

    def test_creates_stage_summary(self, tmp_path: Path) -> None:
        stage_dir = tmp_path / "stage_01"
        stage_dir.mkdir(parents=True, exist_ok=True)
        outcome = ExperimentOutcome(
            timestamp_utc="2025-05-07T12:00:00Z",
            idea_title="Test",
            idea_branch_name="test",
            feature_branch="feat/test_c01",
            idea_outline="outline",
            aider_plan_prompt="plan",
            aider_impl_prompt="impl",
            baseline_commit="abc",
            trial_commit="def",
            job_id="123",
            wandb_project="test/project",
            status="finished",
            runtime_seconds=100,
            unified_metric=0.5,
            metric_histories=None,
            baseline_metric=0.4,
            metric_delta=0.1,
            avg_gpu_util=None,
            avg_gpu_memory=None,
            kept=False,
            summary="summary",
            info_log_excerpt="",
            wandb_log_excerpt="",
            err_log_excerpt="",
        )
        path = _write_stage_summary_markdown(
            stage_dir=stage_dir,
            stage_index=1,
            stage_summary_text="Test summary",
            stage_baseline_metric=0.8,
            merged_feature_branches=["feat/a", "feat/b"],
            skipped_conflict_feature_branches=["feat/c"],
            reset_to_best=True,
            stage_improvement_ideas='{"top_improvements": []}',
            stage_outcomes=[outcome],
        )
        assert path.exists()
        content = path.read_text()
        assert "Stage 0001 Summary" in content


class TestAppendExperimentsMd:
    """Test _append_experiments_md function."""

    def test_creates_parent_dir(self, tmp_path: Path) -> None:
        md_path = tmp_path / "experiments.md"
        outcome = ExperimentOutcome(
            timestamp_utc="2025-05-07T12:00:00Z",
            idea_title="Test",
            idea_branch_name="test",
            feature_branch="feat/test",
            idea_outline="outline",
            aider_plan_prompt="plan",
            aider_impl_prompt="impl",
            baseline_commit="abc",
            trial_commit="def",
            job_id="123",
            wandb_project="test/project",
            status="finished",
            runtime_seconds=100,
            unified_metric=0.5,
            metric_histories=None,
            baseline_metric=0.4,
            metric_delta=0.1,
            avg_gpu_util=None,
            avg_gpu_memory=None,
            kept=False,
            summary="summary",
            info_log_excerpt="",
            wandb_log_excerpt="",
            err_log_excerpt="",
        )
        _append_experiments_md(md_path, outcome)
        assert md_path.exists()

    def test_writes_content(self, tmp_path: Path) -> None:
        md_path = tmp_path / "experiments.md"
        outcome = ExperimentOutcome(
            timestamp_utc="2025-05-07T12:00:00Z",
            idea_title="Test Idea",
            idea_branch_name="test-idea",
            feature_branch="feat/test",
            idea_outline="outline",
            aider_plan_prompt="plan",
            aider_impl_prompt="impl",
            baseline_commit="abc",
            trial_commit="def",
            job_id="123",
            wandb_project="test/project",
            status="finished",
            runtime_seconds=100,
            unified_metric=0.5,
            metric_histories=None,
            baseline_metric=0.4,
            metric_delta=0.1,
            avg_gpu_util=None,
            avg_gpu_memory=None,
            kept=False,
            summary="summary",
            info_log_excerpt="",
            wandb_log_excerpt="",
            err_log_excerpt="",
        )
        _append_experiments_md(md_path, outcome)
        content = md_path.read_text()
        assert "Test Idea" in content
        assert "##" in content


class TestAppendExperimentsTsv:
    """Test _append_experiments_tsv function."""

    def test_creates_header_on_new_file(self, tmp_path: Path) -> None:
        tsv_path = tmp_path / "experiments.tsv"
        outcome = ExperimentOutcome(
            timestamp_utc="2025-05-07T12:00:00Z",
            idea_title="Test",
            idea_branch_name="test",
            feature_branch="feat/test",
            idea_outline="outline",
            aider_plan_prompt="plan",
            aider_impl_prompt="impl",
            baseline_commit="abc",
            trial_commit="def",
            job_id="123",
            wandb_project="test/project",
            status="finished",
            runtime_seconds=100,
            unified_metric=0.5,
            metric_histories=None,
            baseline_metric=0.4,
            metric_delta=0.1,
            avg_gpu_util=None,
            avg_gpu_memory=None,
            kept=False,
            summary="summary",
            info_log_excerpt="",
            wandb_log_excerpt="",
            err_log_excerpt="",
        )
        _append_experiments_tsv(tsv_path, outcome)
        content = tsv_path.read_text()
        assert "timestamp_utc" in content.split("\n")[0]

    def test_appends_rows(self, tmp_path: Path) -> None:
        tsv_path = tmp_path / "experiments.tsv"
        outcome1 = ExperimentOutcome(
            timestamp_utc="2025-05-07T12:00:00Z",
            idea_title="Test1",
            idea_branch_name="test1",
            feature_branch="feat/test1",
            idea_outline="o",
            aider_plan_prompt="p",
            aider_impl_prompt="i",
            baseline_commit="a",
            trial_commit="b",
            job_id="1",
            wandb_project="test/project",
            status="finished",
            runtime_seconds=100,
            unified_metric=0.5,
            metric_histories=None,
            baseline_metric=0.4,
            metric_delta=0.1,
            avg_gpu_util=None,
            avg_gpu_memory=None,
            kept=False,
            summary="s",
            info_log_excerpt="",
            wandb_log_excerpt="",
            err_log_excerpt="",
        )
        outcome2 = ExperimentOutcome(
            timestamp_utc="2025-05-08T12:00:00Z",
            idea_title="Test2",
            idea_branch_name="test2",
            feature_branch="feat/test2",
            idea_outline="o",
            aider_plan_prompt="p",
            aider_impl_prompt="i",
            baseline_commit="a",
            trial_commit="c",
            job_id="2",
            wandb_project="test/project",
            status="finished",
            runtime_seconds=200,
            unified_metric=0.6,
            metric_histories=None,
            baseline_metric=0.4,
            metric_delta=0.2,
            avg_gpu_util=None,
            avg_gpu_memory=None,
            kept=True,
            summary="s",
            info_log_excerpt="",
            wandb_log_excerpt="",
            err_log_excerpt="",
        )
        _append_experiments_tsv(tsv_path, outcome1)
        _append_experiments_tsv(tsv_path, outcome2)
        content = tsv_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 3  # header + 2 rows


class TestIsOutcomeAlreadyLogged:
    """Test _is_outcome_already_logged function."""

    def test_missing_file_returns_false(self, tmp_path: Path) -> None:
        tsv_path = tmp_path / "nonexistent.tsv"
        outcome = ExperimentOutcome(
            timestamp_utc="2025-05-07T12:00:00Z",
            idea_title="Test",
            idea_branch_name="test",
            feature_branch="feat/test",
            idea_outline="o",
            aider_plan_prompt="p",
            aider_impl_prompt="i",
            baseline_commit="abc",
            trial_commit="def",
            job_id="123",
            wandb_project="test/project",
            status="finished",
            runtime_seconds=100,
            unified_metric=0.5,
            metric_histories=None,
            baseline_metric=0.4,
            metric_delta=0.1,
            avg_gpu_util=None,
            avg_gpu_memory=None,
            kept=False,
            summary="s",
            info_log_excerpt="",
            wandb_log_excerpt="",
            err_log_excerpt="",
        )
        result = _is_outcome_already_logged(tsv_path, outcome)
        assert result is False

    def test_not_logged_returns_false(self, tmp_path: Path) -> None:
        tsv_path = tmp_path / "experiments.tsv"
        tsv_path.write_text("timestamp_utc\t...\n")
        outcome = ExperimentOutcome(
            timestamp_utc="2025-05-07T12:00:00Z",
            idea_title="Test",
            idea_branch_name="test",
            feature_branch="feat/other",
            idea_outline="o",
            aider_plan_prompt="p",
            aider_impl_prompt="i",
            baseline_commit="abc",
            trial_commit="def",
            job_id="456",
            wandb_project="test/project",
            status="finished",
            runtime_seconds=100,
            unified_metric=0.5,
            metric_histories=None,
            baseline_metric=0.4,
            metric_delta=0.1,
            avg_gpu_util=None,
            avg_gpu_memory=None,
            kept=False,
            summary="s",
            info_log_excerpt="",
            wandb_log_excerpt="",
            err_log_excerpt="",
        )
        result = _is_outcome_already_logged(tsv_path, outcome)
        assert result is False

    def test_already_logged_returns_true(self, tmp_path: Path) -> None:
        tsv_path = tmp_path / "experiments.tsv"
        tsv_path.write_text(
            "timestamp_utc\tidea_title\tidea_branch_name\tfeature_branch\tbaseline_commit\ttrial_commit\tjob_id\twandb_project\tstatus\truntime_seconds\tunified_metric\tmetric_histories\tbaseline_metric\tmetric_delta\tavg_gpu_util\tavg_gpu_memory\tkept\tsummary\n"
            "2025-05-07\tTest\ttest\tfeat/test\tabc\tdef\t123\ttest/project\tfinished\t100\t0.5\t{}\t0.4\t0.1\t\t\tFalse\tsummary\n"
        )
        outcome = ExperimentOutcome(
            timestamp_utc="2025-05-07T12:00:00Z",
            idea_title="Test",
            idea_branch_name="test",
            feature_branch="feat/test",
            idea_outline="o",
            aider_plan_prompt="p",
            aider_impl_prompt="i",
            baseline_commit="abc",
            trial_commit="def",
            job_id="123",
            wandb_project="test/project",
            status="finished",
            runtime_seconds=100,
            unified_metric=0.5,
            metric_histories=None,
            baseline_metric=0.4,
            metric_delta=0.1,
            avg_gpu_util=None,
            avg_gpu_memory=None,
            kept=False,
            summary="summary",
            info_log_excerpt="",
            wandb_log_excerpt="",
            err_log_excerpt="",
        )
        result = _is_outcome_already_logged(tsv_path, outcome)
        assert result is True