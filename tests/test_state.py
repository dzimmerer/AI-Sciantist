"""Tests for sciantist.state module."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from sciantist.config import ExperimentOutcome
from sciantist.state import (
    _append_leaderboard_entry,
    _build_lock_path,
    _build_leaderboard_path,
    _build_stage_plan_path,
    _build_trial_state_path,
    _build_worker_state_path,
    _deserialize_experiment_outcome,
    _expert_payload_from_spec,
    _file_lock,
    _load_json_state,
    _load_leaderboard,
    _load_stage_plan,
    _load_state,
    _load_trial_run_state,
    _load_worker_info,
    _load_worker_runtime_state,
    _merged_validation_entry_exists,
    _refresh_leaderboard_best_flags,
    _leaderboard_run_id_exists,
    _resolve_output_dir,
    _save_json_state,
    _save_leaderboard,
    _save_stage_plan,
    _save_state,
    _save_trial_run_state,
    _save_worker_info,
    _save_worker_runtime_state,
    _select_best_entry,
    _serialize_experiment_outcome,
    _to_finite_metric,
    _update_run_info,
    _update_worker_info,
    _utc_now_iso,
)


class TestUtcNowIso:
    """Test _utc_now_iso function."""

    def test_returns_string(self) -> None:
        result = _utc_now_iso()
        assert isinstance(result, str)

    def test_is_iso_format(self) -> None:
        result = _utc_now_iso()
        assert "T" in result
        assert "Z" in result or "+" in result


class TestResolveOutputDir:
    """Test _resolve_output_dir function."""

    def test_expands_user(self) -> None:
        result = _resolve_output_dir("~/test")
        assert result.is_absolute()

    def test_resolves_path(self) -> None:
        result = _resolve_output_dir("./test")
        assert result.is_absolute()


class TestLoadSaveState:
    """Test _load_state and _save_state functions."""

    def test_load_missing_returns_empty_dict(self, tmp_path: Path) -> None:
        result = _load_state(tmp_path / "nonexistent.json")
        assert result == {}

    def test_load_invalid_json_returns_empty_dict(self, tmp_path: Path) -> None:
        path = tmp_path / "invalid.json"
        path.write_text("not valid json {")
        result = _load_state(path)
        assert result == {}

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        state = {"key": "value", "number": 42}
        _save_state(path, state)
        result = _load_state(path)
        assert result == state


class TestLoadSaveJsonState:
    """Test _load_json_state and _save_json_state functions."""

    def test_load_missing_returns_empty_dict(self, tmp_path: Path) -> None:
        result = _load_json_state(tmp_path / "nonexistent.json")
        assert result == {}

    def test_load_invalid_json_returns_empty_dict(self, tmp_path: Path) -> None:
        path = tmp_path / "invalid.json"
        path.write_text("not json")
        result = _load_json_state(path)
        assert result == {}

    def test_load_non_dict_returns_empty_dict(self, tmp_path: Path) -> None:
        path = tmp_path / "list.json"
        path.write_text("[1, 2, 3]")
        result = _load_json_state(path)
        assert result == {}

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "data.json"
        data = {"key": "value"}
        _save_json_state(path, data)
        result = _load_json_state(path)
        assert result == data

    def test_atomic_write(self, tmp_path: Path) -> None:
        path = tmp_path / "atomic.json"
        data = {"test": "value"}
        _save_json_state(path, data)
        assert path.exists()
        result = _load_json_state(path)
        assert result == data


class TestTrialState:
    """Test trial run state functions."""

    def test_build_trial_state_path(self, tmp_path: Path) -> None:
        result = _build_trial_state_path(tmp_path / "candidate1")
        assert result.name == "run_state.json"

    def test_save_load_trial_run_state(self, tmp_path: Path) -> None:
        path = tmp_path / "run_state.json"
        state = {"stage_index": 1, "candidate_label": "test"}
        _save_trial_run_state(path, state)
        result = _load_trial_run_state(path)
        assert result["stage_index"] == 1
        assert "checkpoint_updated_ts_utc" in result

    def test_load_missing_trial_returns_empty_dict(self, tmp_path: Path) -> None:
        result = _load_trial_run_state(tmp_path / "nonexistent.json")
        assert result == {}


class TestStagePlan:
    """Test stage plan functions."""

    def test_build_stage_plan_path(self, tmp_path: Path) -> None:
        result = _build_stage_plan_path(tmp_path / "stage1")
        assert result.name == "stage_plan.json"

    def test_save_load_stage_plan(self, tmp_path: Path) -> None:
        path = tmp_path / "plan.json"
        plan = {"stage_index": 2, "ideas": ["idea1", "idea2"]}
        _save_stage_plan(path, plan)
        result = _load_stage_plan(path)
        assert result == plan

    def test_load_missing_stage_plan_returns_empty_dict(self, tmp_path: Path) -> None:
        result = _load_stage_plan(tmp_path / "nonexistent.json")
        assert result == {}


class TestLockPath:
    """Test lock path building."""

    def test_build_lock_path(self, tmp_path: Path) -> None:
        result = _build_lock_path(tmp_path, "test.lock")
        assert result.name == "test.lock"
        assert result.parent == tmp_path


class TestFileLock:
    """Test _file_lock context manager."""

    def test_acquires_and_releases_lock(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "lock.txt"
        with _file_lock(lock_path):
            assert lock_path.exists()
        assert lock_path.exists()

    def test_nested_lock_with_different_files_succeeds(self, tmp_path: Path) -> None:
        lock_path1 = tmp_path / "lock1.txt"
        lock_path2 = tmp_path / "lock2.txt"
        with _file_lock(lock_path1):
            with _file_lock(lock_path2):
                pass


class TestToFiniteMetric:
    """Test _to_finite_metric function."""

    def test_integer_returns_float(self) -> None:
        assert _to_finite_metric(42) == 42.0

    def test_float_returns_float(self) -> None:
        assert _to_finite_metric(3.14) == 3.14

    def test_none_returns_none(self) -> None:
        assert _to_finite_metric(None) is None

    def test_string_returns_none(self) -> None:
        assert _to_finite_metric("not a number") is None

    def test_bool_returns_none(self) -> None:
        assert _to_finite_metric(True) is None
        assert _to_finite_metric(False) is None

    def test_inf_returns_none(self) -> None:
        assert _to_finite_metric(float("inf")) is None
        assert _to_finite_metric(float("-inf")) is None

    def test_nan_returns_none(self) -> None:
        assert _to_finite_metric(float("nan")) is None


class TestLeaderboard:
    """Test leaderboard functions."""

    def test_build_leaderboard_path(self, tmp_path: Path) -> None:
        result = _build_leaderboard_path(tmp_path, "leaderboard.json")
        assert result.name == "leaderboard.json"

    def test_load_empty_leaderboard(self, tmp_path: Path) -> None:
        path = tmp_path / "leaderboard.json"
        result = _load_leaderboard(path)
        assert result == []

    def test_load_invalid_leaderboard(self, tmp_path: Path) -> None:
        path = tmp_path / "leaderboard.json"
        path.write_text("not a list")
        result = _load_leaderboard(path)
        assert result == []

    def test_save_load_leaderboard(self, tmp_path: Path) -> None:
        path = tmp_path / "leaderboard.json"
        entries = [{"run_id": "1", "unified_metric": 0.5}, {"run_id": "2", "unified_metric": 0.8}]
        _save_leaderboard(path, entries)
        result = _load_leaderboard(path)
        assert len(result) == 2

    def test_leaderboard_filters_non_dict_entries(self, tmp_path: Path) -> None:
        path = tmp_path / "leaderboard.json"
        path.write_text('[{"run_id": "1"}, "string", null, {"run_id": "2"}]')
        result = _load_leaderboard(path)
        assert len(result) == 2


class TestSelectBestEntry:
    """Test _select_best_entry function."""

    def test_empty_list_returns_none(self) -> None:
        assert _select_best_entry([], higher_is_better=True) is None

    def test_no_finished_returns_none(self) -> None:
        entries = [{"status": "running", "unified_metric": 0.9}]
        assert _select_best_entry(entries, higher_is_better=True) is None

    def test_no_metric_returns_none(self) -> None:
        entries = [{"status": "finished"}]
        assert _select_best_entry(entries, higher_is_better=True) is None

    def test_selects_highest_when_higher_is_better(self) -> None:
        entries = [
            {"status": "finished", "unified_metric": 0.5},
            {"status": "finished", "unified_metric": 0.9},
            {"status": "finished", "unified_metric": 0.7},
        ]
        result = _select_best_entry(entries, higher_is_better=True)
        assert result["unified_metric"] == 0.9

    def test_selects_lowest_when_lower_is_better(self) -> None:
        entries = [
            {"status": "finished", "unified_metric": 0.5},
            {"status": "finished", "unified_metric": 0.9},
            {"status": "finished", "unified_metric": 0.7},
        ]
        result = _select_best_entry(entries, higher_is_better=False)
        assert result["unified_metric"] == 0.5

    def test_ignores_non_finished(self) -> None:
        entries = [
            {"status": "crashed", "unified_metric": 0.95},
            {"status": "finished", "unified_metric": 0.8},
        ]
        result = _select_best_entry(entries, higher_is_better=True)
        assert result["unified_metric"] == 0.8


class TestAppendLeaderboardEntry:
    """Test _append_leaderboard_entry function."""

    def test_empty_run_id_returns_false(self, tmp_path: Path) -> None:
        leaderboard_path = tmp_path / "leaderboard.json"
        lock_path = tmp_path / "lock.txt"
        entry = {"run_id": "", "unified_metric": 0.5}
        result = _append_leaderboard_entry(leaderboard_path, lock_path, entry)
        assert result is False

    def test_append_first_entry(self, tmp_path: Path) -> None:
        leaderboard_path = tmp_path / "leaderboard.json"
        lock_path = tmp_path / "lock.txt"
        entry = {"run_id": "run1", "unified_metric": 0.5, "timestamp_utc": "2025-01-01"}
        result = _append_leaderboard_entry(leaderboard_path, lock_path, entry)
        assert result is True
        leaderboard = _load_leaderboard(leaderboard_path)
        assert len(leaderboard) == 1

    def test_duplicate_run_id_returns_false(self, tmp_path: Path) -> None:
        leaderboard_path = tmp_path / "leaderboard.json"
        lock_path = tmp_path / "lock.txt"
        entry1 = {"run_id": "run1", "unified_metric": 0.5, "timestamp_utc": "2025-01-01"}
        entry2 = {"run_id": "run1", "unified_metric": 0.6, "timestamp_utc": "2025-01-02"}
        _append_leaderboard_entry(leaderboard_path, lock_path, entry1)
        result = _append_leaderboard_entry(leaderboard_path, lock_path, entry2)
        assert result is False

    def test_entries_sorted_by_metric(self, tmp_path: Path) -> None:
        leaderboard_path = tmp_path / "leaderboard.json"
        lock_path = tmp_path / "lock.txt"
        _append_leaderboard_entry(leaderboard_path, lock_path, {"run_id": "low", "unified_metric": 0.3, "timestamp_utc": "2025-01-01"})
        _append_leaderboard_entry(leaderboard_path, lock_path, {"run_id": "high", "unified_metric": 0.9, "timestamp_utc": "2025-01-01"})
        _append_leaderboard_entry(leaderboard_path, lock_path, {"run_id": "mid", "unified_metric": 0.6, "timestamp_utc": "2025-01-01"})
        leaderboard = _load_leaderboard(leaderboard_path)
        assert leaderboard[0]["run_id"] == "high"
        assert leaderboard[-1]["run_id"] == "low"


class TestRefreshLeaderboardBestFlags:
    """Test _refresh_leaderboard_best_flags function."""

    def test_no_finished_entries(self, tmp_path: Path) -> None:
        leaderboard_path = tmp_path / "leaderboard.json"
        lock_path = tmp_path / "lock.txt"
        _save_leaderboard(leaderboard_path, [{"status": "running", "run_id": "1"}])
        result = _refresh_leaderboard_best_flags(leaderboard_path, lock_path, True)
        assert result is None

    def test_marks_best_path_flag(self, tmp_path: Path) -> None:
        leaderboard_path = tmp_path / "leaderboard.json"
        lock_path = tmp_path / "lock.txt"
        entries = [
            {"run_id": "run1", "status": "finished", "unified_metric": 0.5, "currently_in_best_path": False, "parent_entry_ids": []},
            {"run_id": "run2", "status": "finished", "unified_metric": 0.9, "currently_in_best_path": False, "parent_entry_ids": []},
        ]
        _save_leaderboard(leaderboard_path, entries)
        best = _refresh_leaderboard_best_flags(leaderboard_path, lock_path, True)
        assert best["run_id"] == "run2"
        leaderboard = _load_leaderboard(leaderboard_path)
        best_entry = next(e for e in leaderboard if e["run_id"] == "run2")
        assert best_entry["currently_in_best_path"] is True


class TestWorkerState:
    """Test worker state functions."""

    def test_build_worker_state_path(self, tmp_path: Path) -> None:
        result = _build_worker_state_path(tmp_path, "worker_01")
        assert result.name == "worker_state.json"
        assert "worker_01" in str(result)

    def test_save_load_worker_runtime_state(self, tmp_path: Path) -> None:
        path = tmp_path / "worker_state.json"
        state = {"run_index": 5, "active_cycle": {"stage": "executing"}}
        _save_worker_runtime_state(path, state)
        result = _load_worker_runtime_state(path)
        assert result["run_index"] == 5
        assert "checkpoint_updated_ts_utc" in result

    def test_load_missing_worker_state_returns_empty_dict(self, tmp_path: Path) -> None:
        result = _load_worker_runtime_state(tmp_path / "nonexistent.json")
        assert result == {}

    def test_save_load_worker_info(self, tmp_path: Path) -> None:
        path = tmp_path / "worker_info.json"
        payload = {"expert": {"name": "Test", "description": "Test expert"}, "current_run_index": 1}
        _save_worker_info(path, payload)
        result = _load_worker_info(path)
        assert result["expert"]["name"] == "Test"

    def test_load_missing_worker_info_returns_empty_dict(self, tmp_path: Path) -> None:
        result = _load_worker_info(tmp_path / "nonexistent.json")
        assert result == {}


class TestUpdateWorkerInfo:
    """Test _update_worker_info function."""

    def test_creates_new_info(self, tmp_path: Path) -> None:
        info_path = tmp_path / "worker_info.json"
        _update_worker_info(info_path, expert_spec=("Test Expert", "A test"), current_run_index=1, run_id="run1")
        result = _load_worker_info(info_path)
        assert result["current_run_index"] == 1
        assert result["expert"]["name"] == "Test Expert"
        assert "run1" in result["previous_run_ids"]

    def test_accumulates_run_ids(self, tmp_path: Path) -> None:
        info_path = tmp_path / "worker_info.json"
        _update_worker_info(info_path, expert_spec=None, current_run_index=1, run_id="run1")
        _update_worker_info(info_path, expert_spec=None, current_run_index=2, run_id="run2")
        result = _load_worker_info(info_path)
        assert "run1" in result["previous_run_ids"]
        assert "run2" in result["previous_run_ids"]


class TestUpdateRunInfo:
    """Test _update_run_info function."""

    def test_updates_existing_info(self, tmp_path: Path) -> None:
        info_path = tmp_path / "run_info.json"
        _save_worker_info(info_path, {"worker_id": "worker_01"})
        _update_run_info(info_path, {"status": "executing", "run_index": 5})
        result = _load_worker_info(info_path)
        assert result["status"] == "executing"
        assert result["run_index"] == 5


class TestExpertPayloadFromSpec:
    """Test _expert_payload_from_spec function."""

    def test_none_returns_none(self) -> None:
        assert _expert_payload_from_spec(None) is None

    def test_tuple_returns_dict(self) -> None:
        result = _expert_payload_from_spec(("Test Expert", "A test description"))
        assert result == {"name": "Test Expert", "description": "A test description"}


class TestSerializeDeserializeExperimentOutcome:
    """Test _serialize_experiment_outcome and _deserialize_experiment_outcome."""

    def test_roundtrip(self) -> None:
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
            runtime_seconds=3600,
            unified_metric=0.85,
            metric_histories={"val/x": [0.5, 0.85]},
            baseline_metric=0.80,
            metric_delta=0.05,
            avg_gpu_util=50.0,
            avg_gpu_memory=70.0,
            kept=True,
            summary="summary",
            info_log_excerpt="info",
            wandb_log_excerpt="wandb",
            err_log_excerpt="err",
        )
        serialized = _serialize_experiment_outcome(outcome)
        assert isinstance(serialized, dict)
        deserialized = _deserialize_experiment_outcome(serialized)
        assert deserialized is not None
        assert deserialized.idea_title == "Test"
        assert deserialized.unified_metric == 0.85

    def test_deserialize_non_dict_returns_none(self) -> None:
        result = _deserialize_experiment_outcome("not a dict")
        assert result is None

    def test_deserialize_invalid_dict_returns_none(self) -> None:
        result = _deserialize_experiment_outcome({"unknown_field": "value"})
        assert result is None


class TestLeaderboardRunIdExists:
    """Test _leaderboard_run_id_exists function."""

    def test_empty_leaderboard_returns_false(self, tmp_path: Path) -> None:
        leaderboard_path = tmp_path / "leaderboard.json"
        _save_leaderboard(leaderboard_path, [])
        result = _leaderboard_run_id_exists(leaderboard_path, "run1")
        assert result is False

    def test_missing_run_id_returns_false(self, tmp_path: Path) -> None:
        leaderboard_path = tmp_path / "leaderboard.json"
        _save_leaderboard(leaderboard_path, [{"run_id": "run1", "unified_metric": 0.5}])
        result = _leaderboard_run_id_exists(leaderboard_path, "run2")
        assert result is False

    def test_existing_run_id_returns_true(self, tmp_path: Path) -> None:
        leaderboard_path = tmp_path / "leaderboard.json"
        _save_leaderboard(leaderboard_path, [{"run_id": "run1", "unified_metric": 0.5}])
        result = _leaderboard_run_id_exists(leaderboard_path, "run1")
        assert result is True

    def test_empty_run_id_returns_false(self, tmp_path: Path) -> None:
        leaderboard_path = tmp_path / "leaderboard.json"
        _save_leaderboard(leaderboard_path, [{"run_id": "run1", "unified_metric": 0.5}])
        result = _leaderboard_run_id_exists(leaderboard_path, "")
        assert result is False


class TestMergedValidationEntryExists:
    """Test _merged_validation_entry_exists function."""

    def test_no_merged_validation_returns_false(self, tmp_path: Path) -> None:
        leaderboard_path = tmp_path / "leaderboard.json"
        _save_leaderboard(leaderboard_path, [{"run_id": "run1"}])
        result = _merged_validation_entry_exists(leaderboard_path, "merge_run1")
        assert result is False

    def test_with_merged_validation_returns_true(self, tmp_path: Path) -> None:
        leaderboard_path = tmp_path / "leaderboard.json"
        _save_leaderboard(leaderboard_path, [{"run_id": "run1", "merged_validation_of": "merge_run1"}])
        result = _merged_validation_entry_exists(leaderboard_path, "merge_run1")
        assert result is True