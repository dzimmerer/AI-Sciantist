from __future__ import annotations

import json
from pathlib import Path

import pytest

import scian
from sciantist.config import LoopConfig
from sciantist.wandb_ops import _find_run_by_name, create_tracking_backend


class _MockTrackingBackend:
    def __init__(self, payload_sequence: list[dict] | None = None, error_sequence: list[Exception] | None = None) -> None:
        self._payload_sequence = payload_sequence or []
        self._error_sequence = error_sequence or []
        self.fetch_calls = 0
        self.sync_calls = 0

    def sync_from_cluster(self, log_fn=None) -> None:
        self.sync_calls += 1
        if log_fn is not None:
            log_fn("mock sync")

    def fetch_run_snapshot_by_job_id(
        self,
        project_path: str,
        job_id: str,
        output_dir: Path,
        metric_names: list[str] | None = None,
        max_wandb_retries: int = 20,
    ) -> dict:
        del project_path, output_dir, metric_names, max_wandb_retries
        self.fetch_calls += 1
        if self._error_sequence:
            raise self._error_sequence.pop(0)
        if self._payload_sequence:
            return self._payload_sequence.pop(0)
        raise RuntimeError(f"No mock payload available for {job_id}")

    def load_output_log_excerpt(self, payload: dict, max_chars: int = 8000) -> str:
        del payload, max_chars
        return "mock-wandb-log"

    def fetch_avg_gpu_metrics_from_events(self, project_path: str, job_id: str) -> tuple[float | None, float | None]:
        del project_path, job_id
        return 12.5, 33.3


def _base_config() -> LoopConfig:
    return LoopConfig(
        target_repo="/tmp/repo",
        original_target_repo="/tmp/repo",
        train_command="train",
        runtime="00:01:00",
        poll_seconds=1,
        max_fix_attempts=1,
        max_wandb_retries=3,
        tracking_backend="wandb",
        wandb_project="entity/project",
        metric_weights={"val/x": 1.0},
        verbose=False,
    )


def test_create_tracking_backend_from_config_name() -> None:
    backend = create_tracking_backend("wandb")
    assert backend.__class__.__name__ == "WandbTrackingBackend"


def test_create_tracking_backend_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported tracking backend"):
        create_tracking_backend("unknown")


def test_run_on_cluster_success_with_mock_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _base_config()
    backend = _MockTrackingBackend(
        payload_sequence=[
            {
                "summary": {"val/x": 0.9},
                "metric_histories": {"val/x": [0.5, 0.9]},
                "last_step_metrics": {"val/x": 0.9},
                "downloaded_log_files": [],
            }
        ]
    )

    monkeypatch.setattr(scian, "create_tracking_backend", lambda backend_name: backend)
    monkeypatch.setattr(scian, "_resolve_train_command_for_repo", lambda *args, **kwargs: "train")
    monkeypatch.setattr(scian, "submit_cluster_job", lambda *args, **kwargs: 101)
    monkeypatch.setattr(scian, "_poll_job_until_terminal", lambda *args, **kwargs: ("finished", 17))
    monkeypatch.setattr(scian, "load_remote_log", lambda *args, **kwargs: "log")

    trial_state_path = tmp_path / "run_state.json"
    result = scian._run_on_cluster(
        config=config,
        files_to_edit=[],
        idea_title="idea",
        skip_aider=True,
        trial_state_path=trial_state_path,
        wandb_output_dir=tmp_path / "wandb_pull",
        expected_trial_commit="trial-commit",
    )

    assert result[0] == "finished"
    assert result[2] == "101"
    assert result[5] == "mock-wandb-log"
    assert result[6] == pytest.approx(0.9)
    assert result[7] == pytest.approx(12.5)
    assert result[8] == pytest.approx(33.3)
    assert result[9]["val/x"] == [0.5, 0.9]
    assert backend.sync_calls == 1
    assert backend.fetch_calls == 1

    run_state = json.loads(trial_state_path.read_text(encoding="utf-8"))
    assert run_state.get("wandb_fetched") is True
    assert run_state.get("wandb_missing_run") is False


def test_run_on_cluster_missing_run_triggers_recovery_and_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _base_config()
    backend = _MockTrackingBackend(
        payload_sequence=[
            {
                "summary": {"val/x": 0.5},
                "metric_histories": {"val/x": [0.5]},
                "last_step_metrics": {"val/x": 0.5},
                "downloaded_log_files": [],
            }
        ],
        error_sequence=[RuntimeError("No run named '101' found in project 'entity/project'.")],
    )

    submitted_ids = iter([101, 202])
    monkeypatch.setattr(scian, "create_tracking_backend", lambda backend_name: backend)
    monkeypatch.setattr(scian, "_resolve_train_command_for_repo", lambda *args, **kwargs: "train")
    monkeypatch.setattr(scian, "submit_cluster_job", lambda *args, **kwargs: next(submitted_ids))
    monkeypatch.setattr(scian, "_poll_job_until_terminal", lambda *args, **kwargs: ("finished", 22))
    monkeypatch.setattr(scian, "load_remote_log", lambda *args, **kwargs: "log")
    monkeypatch.setattr(scian, "_try_fix_crash_with_aider", lambda *args, **kwargs: "fixed-commit")

    trial_state_path = tmp_path / "run_state.json"
    result = scian._run_on_cluster(
        config=config,
        files_to_edit=["train.py"],
        idea_title="idea",
        skip_aider=False,
        trial_state_path=trial_state_path,
        wandb_output_dir=tmp_path / "wandb_pull",
        expected_trial_commit="trial-commit",
    )

    assert result[0] == "finished"
    assert result[2] == "202"
    assert result[6] == pytest.approx(0.5)
    assert backend.fetch_calls == 2


def test_find_run_by_name_retries_transient_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class CommError(Exception):
        pass

    class _FakeRun:
        def __init__(self, run_id: str) -> None:
            self.id = run_id
            self.name = run_id
            self.display_name = run_id

    class _FakeApi:
        def __init__(self) -> None:
            self.calls = 0

        def runs(self, project_path: str):
            del project_path
            self.calls += 1
            if self.calls == 1:
                raise CommError("timed out")
            return [_FakeRun("job-123")]

    monkeypatch.setattr("sciantist.wandb_ops.time.sleep", lambda seconds: None)

    api = _FakeApi()
    run = _find_run_by_name(
        api=api,
        project_path="entity/project",
        run_name="job-123",
        max_wandb_retries=2,
        initial_retry_timeout_seconds=1,
    )

    assert run.id == "job-123"
    assert api.calls == 2
